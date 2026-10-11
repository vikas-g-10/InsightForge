"""InsightForge backend: FastAPI application (Milestone 1 foundation)."""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from auth import AuthenticatedUser, get_current_user
from ingest import IngestError, file_too_large, ingest_file, read_upload
from pipeline.analyze import analyze
from pipeline.compare import compare_products, compare_versions, get_available_targets
from pipeline.emerging import detect_emerging_issues
from pipeline.prioritization import simulate_fixes
from pipeline.root_cause import batch_generate_hypotheses, generate_root_cause_hypotheses
from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AvailableTargetsRequest,
    AvailableTargetsResponse,
    EmergingIssuesRequest,
    EmergingIssuesResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ImageDefectAnalysis,
    IngestMetadata,
    IssuesResponse,
    MLMetricsResponse,
    MLPredictRequest,
    MLPredictResponse,
    ProductCompareRequest,
    ProductCompareResponse,
    RootCauseAnalysisResult,
    RootCauseBatchResponse,
    RootCauseRequest,
    SimulateRequest,
    SimulateResponse,
    VersionCompareRequest,
    VersionCompareResponse,
    VisionHealthResponse,
)
from ml.model_service import ModelArtifactNotFoundError, SentimentModelService
from services.vision import VisionError, VisionService
from settings import Settings


logger = logging.getLogger("insightforge")

_HTTP_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
}

# Allowance for multipart framing when comparing Content-Length to the file limit.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_INGEST_ERROR_RESPONSES = {
    status: {"model": ErrorResponse} for status in (400, 413, 415, 422)
}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    columns: list[str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, columns=columns))
    # exclude_none keeps the original {"code","message"} shape when no columns.
    return JSONResponse(
        body.model_dump(exclude_none=True), status_code=status_code, headers=headers
    )


def create_app(
    settings: Settings | None = None,
    vision_service: VisionService | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    vision_service = vision_service or VisionService(settings)
    app = FastAPI(title="InsightForge API", version="0.1.0")
    app.state.settings = settings

    async def reject_oversized_body(request: Request, call_next):
        """Cheap early 413 using Content-Length, before the body is parsed."""
        length = request.headers.get("content-length", "")
        is_vision = request.url.path.startswith("/api/vision")
        limit_bytes = settings.max_image_upload_bytes if is_vision else settings.max_upload_bytes
        limit = limit_bytes + _MULTIPART_OVERHEAD_BYTES
        if request.method == "POST" and length.isdigit() and int(length) > limit:
            err = file_too_large(limit_bytes)
            return _error_response(err.status_code, err.code, err.message)
        return await call_next(request)

    # Added before CORS so CORS stays outermost and 413s still carry CORS headers.
    app.add_middleware(BaseHTTPMiddleware, dispatch=reject_oversized_body)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,  # no auth in this version; never combine with "*"
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code in (401, 403) and exc.detail:
            message = str(exc.detail)
            logger.warning(
                "Auth rejection on %s %s (status %d): %s",
                request.method,
                request.url.path,
                exc.status_code,
                exc.detail,
            )
        else:
            try:
                message = HTTPStatus(exc.status_code).phrase
            except ValueError:
                message = "Request failed"
        code = _HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return _error_response(exc.status_code, code, message, exc.headers)

    @app.exception_handler(IngestError)
    async def ingest_error_handler(request: Request, exc: IngestError):
        logger.info("Upload rejected code=%s", exc.code)
        return _error_response(
            exc.status_code, exc.code, exc.message, columns=exc.columns
        )

    @app.exception_handler(VisionError)
    async def vision_error_handler(request: Request, exc: VisionError):
        logger.info("Vision operation rejected code=%s: %s", exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)


    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Pydantic details can echo submitted input, so they are not returned.
        return _error_response(422, "validation_error", "Request validation failed.")

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled error on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return _error_response(500, "internal_error", "An unexpected error occurred.")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", mode=settings.mode, db_connected=False)

    @app.post(
        "/api/ingest/validate",
        response_model=IngestMetadata,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def validate_upload(
        file: UploadFile = File(...),
        text_column: str | None = Form(default=None),
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> IngestMetadata:
        """Validate a CSV/XLSX review file and report metadata (no analysis)."""
        try:
            data = await read_upload(file, settings.max_upload_bytes)
        finally:
            await file.close()
        result = await run_in_threadpool(
            ingest_file,
            file.filename or "",
            data,
            max_rows=settings.max_rows,
            text_column=text_column,
        )
        return result.metadata

    @app.post(
        "/api/analyze",
        response_model=AnalyzeResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def analyze_endpoint(
        request: AnalyzeRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> AnalyzeResponse:
        """Run the core review analysis pipeline on a batch of reviews."""
        if not request.reviews:
            raise IngestError("empty_batch", "The reviews list must not be empty.", 400)
        if len(request.reviews) > settings.max_rows:
            raise IngestError(
                "payload_too_large",
                f"Batch exceeds the {settings.max_rows} review limit.",
                413,
            )
        result = await run_in_threadpool(
            analyze,
            request.reviews,
            request.mode,
            max_rows=settings.max_rows,
        )
        return result

    @app.post(
        "/api/analyze/file",
        response_model=AnalyzeResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def analyze_file_upload(
        file: UploadFile = File(...),
        text_column: str | None = Form(default=None),
        mode: str = Form(default="lite"),
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> AnalyzeResponse:
        """Upload and analyze a CSV/XLSX review file."""
        try:
            data = await read_upload(file, settings.max_upload_bytes)
        finally:
            await file.close()
        ingest_res = await run_in_threadpool(
            ingest_file,
            file.filename or "",
            data,
            max_rows=settings.max_rows,
            text_column=text_column,
        )
        analysis_res = await run_in_threadpool(
            analyze,
            ingest_res.reviews,
            mode,
            max_rows=settings.max_rows,
        )
        return analysis_res

    @app.post(
        "/api/issues",
        response_model=IssuesResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def get_issues_endpoint(
        request: AnalyzeRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> IssuesResponse:
        """Analyze reviews, cluster clauses, and return prioritized issue insights."""
        if not request.reviews:
            raise IngestError("empty_batch", "The reviews list must not be empty.", 400)
        if len(request.reviews) > settings.max_rows:
            raise IngestError(
                "payload_too_large",
                f"Batch exceeds the {settings.max_rows} review limit.",
                413,
            )
        analysis_res = await run_in_threadpool(
            analyze,
            request.reviews,
            request.mode,
            max_rows=settings.max_rows,
        )
        return IssuesResponse(
            clusters=analysis_res.clusters,
            insights=analysis_res.insights,
            total_clusters=len(analysis_res.clusters),
            top_aspects=[ins.aspect for ins in analysis_res.insights[:5]],
            warnings=analysis_res.meta.summary.warnings,
        )

    @app.post(
        "/api/simulate",
        response_model=SimulateResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def simulate_endpoint(
        request: SimulateRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> SimulateResponse:
        """Run the Fix-This-First simulator for proposed aspect fix percentages."""
        if not request.reviews:
            raise IngestError("empty_batch", "Reviews are required to run simulation.", 400)
        analysis_res = await run_in_threadpool(
            analyze,
            request.reviews,
            "lite",
            max_rows=settings.max_rows,
        )
        sim_res = simulate_fixes(
            analysis_res.reviews,
            analysis_res.insights,
            request.fixes,
        )
        return sim_res

    @app.post(
        "/api/emerging-issues",
        response_model=EmergingIssuesResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def emerging_issues_endpoint(
        request: EmergingIssuesRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> EmergingIssuesResponse:
        """Detect new, accelerating, and safety-escalated issues across time windows."""
        if not request.reviews:
            raise IngestError("empty_batch", "The reviews list must not be empty.", 400)
        if len(request.reviews) > settings.max_rows:
            raise IngestError(
                "payload_too_large",
                f"Batch exceeds the {settings.max_rows} review limit.",
                413,
            )
        analysis_res = await run_in_threadpool(
            analyze,
            request.reviews,
            request.mode,
            max_rows=settings.max_rows,
        )
        emerging_res = detect_emerging_issues(
            analysis_res.reviews,
            analysis_res.clusters,
            recent_days=request.recent_days,
            baseline_days=request.baseline_days,
            split_date=request.split_date,
            min_recent_count=request.min_recent_count,
        )
        return emerging_res

    @app.get("/api/vision/health", response_model=VisionHealthResponse)
    async def vision_health_endpoint(
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> VisionHealthResponse:
        """Verify Ollama availability and model vision capabilities."""
        return await run_in_threadpool(vision_service.check_health)

    @app.post(
        "/api/vision/analyze",
        response_model=ImageDefectAnalysis,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def vision_analyze_endpoint(
        file: UploadFile = File(...),
        cluster_context: str | None = Form(default=None),
        aspect_context: str | None = Form(default=None),
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> ImageDefectAnalysis:
        """Analyze uploaded product image for defect evidence using Ollama vision."""
        try:
            image_bytes = await file.read()
        finally:
            await file.close()

        return await run_in_threadpool(
            vision_service.analyze_image,
            image_bytes=image_bytes,
            filename=file.filename or "image.jpg",
            declared_mime=file.content_type,
            cluster_context=cluster_context,
            aspect_context=aspect_context,
        )

    @app.post(
        "/api/root-cause/hypotheses",
        response_model=RootCauseBatchResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def root_cause_hypotheses_endpoint(
        request: RootCauseRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> RootCauseBatchResponse:
        """Deduce evidence-grounded engineering failure hypotheses with falsification criteria."""
        if not request.reviews:
            raise IngestError("empty_batch", "The reviews list must not be empty.", 400)
        if len(request.reviews) > settings.max_rows:
            raise IngestError(
                "payload_too_large",
                f"Batch exceeds the {settings.max_rows} review limit.",
                413,
            )

        analysis_res = await run_in_threadpool(
            analyze,
            request.reviews,
            request.mode,
            max_rows=settings.max_rows,
        )

        batch_res = await run_in_threadpool(
            batch_generate_hypotheses,
            clusters=analysis_res.clusters,
            all_reviews=analysis_res.reviews,
            image_findings=request.image_findings,
            target_cluster_id=request.cluster_id,
        )
        return batch_res

    @app.post(
        "/api/compare/targets",
        response_model=AvailableTargetsResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def compare_targets_endpoint(
        request: AvailableTargetsRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> AvailableTargetsResponse:
        """Inspect review batch and return available products and versions for comparison."""
        return await run_in_threadpool(get_available_targets, request.reviews)

    @app.post(
        "/api/compare/versions",
        response_model=VersionCompareResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def compare_versions_endpoint(
        request: VersionCompareRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> VersionCompareResponse:
        """Compare two versions of the same product with grounded metrics, cluster deltas, and evidence."""
        if request.reviews_a is not None and request.reviews_b is not None:
            if not request.reviews_a and not request.reviews_b:
                raise IngestError("empty_batch", "Reviews lists for both versions must not be empty.", 400)
            combined: list[ReviewInputItem] = []
            for item in request.reviews_a:
                cloned = item.model_copy()
                cloned.version = request.version_a
                if request.product and not cloned.product:
                    cloned.product = request.product
                combined.append(cloned)
            for item in request.reviews_b:
                cloned = item.model_copy()
                cloned.version = request.version_b
                if request.product and not cloned.product:
                    cloned.product = request.product
                combined.append(cloned)
            reviews_to_process = combined
        elif request.reviews:
            reviews_to_process = request.reviews
        else:
            raise IngestError("empty_batch", "Must provide either 'reviews' or both 'reviews_a' and 'reviews_b'.", 400)

        if len(reviews_to_process) > settings.max_rows:
            raise IngestError("payload_too_large", f"Batch exceeds the {settings.max_rows} review limit.", 413)

        return await run_in_threadpool(
            compare_versions,
            reviews_to_process,
            request.version_a,
            request.version_b,
            product=request.product,
            mode=request.mode,
        )

    @app.post(
        "/api/compare/products",
        response_model=ProductCompareResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def compare_products_endpoint(
        request: ProductCompareRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> ProductCompareResponse:
        """Compare two distinct products with aspect-by-aspect strengths, common themes, and trade-off analysis."""
        if request.reviews_a is not None and request.reviews_b is not None:
            if not request.reviews_a and not request.reviews_b:
                raise IngestError("empty_batch", "Reviews lists for both products must not be empty.", 400)
            combined = []
            for item in request.reviews_a:
                cloned = item.model_copy()
                cloned.product = request.product_a
                combined.append(cloned)
            for item in request.reviews_b:
                cloned = item.model_copy()
                cloned.product = request.product_b
                combined.append(cloned)
            reviews_to_process = combined
        elif request.reviews:
            reviews_to_process = request.reviews
        else:
            raise IngestError("empty_batch", "Must provide either 'reviews' or both 'reviews_a' and 'reviews_b'.", 400)

        if len(reviews_to_process) > settings.max_rows:
            raise IngestError("payload_too_large", f"Batch exceeds the {settings.max_rows} review limit.", 413)

        return await run_in_threadpool(
            compare_products,
            reviews_to_process,
            request.product_a,
            request.product_b,
            focus_aspect=request.focus_aspect,
            mode=request.mode,
        )

    @app.get(
        "/api/ml/metrics",
        response_model=MLMetricsResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def get_ml_metrics_endpoint(
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> MLMetricsResponse:
        """Fetch empirical evaluation metrics from held-out Amazon Reviews 2023 test set."""
        service = SentimentModelService.get_instance()
        try:
            metrics = await run_in_threadpool(service.get_metrics)
            return MLMetricsResponse(**metrics)
        except ModelArtifactNotFoundError as e:
            raise IngestError("model_not_found", str(e), 404)

    @app.post(
        "/api/ml/predict",
        response_model=MLPredictResponse,
        responses=_INGEST_ERROR_RESPONSES,
    )
    async def predict_sentiment_endpoint(
        request: MLPredictRequest,
        _user: AuthenticatedUser = Depends(get_current_user),
    ) -> MLPredictResponse:
        """Predict sentiment and class probabilities using the trained model pipeline."""
        service = SentimentModelService.get_instance()
        try:
            result = await run_in_threadpool(service.predict, request.text)
            return MLPredictResponse(**result)
        except ModelArtifactNotFoundError as e:
            raise IngestError("model_not_found", str(e), 503)
        except ValueError as e:
            raise IngestError("bad_request", str(e), 400)

    return app


app = create_app()


