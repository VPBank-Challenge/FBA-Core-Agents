from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import setup_logger
from typing import Dict
from datetime import datetime

from src.optimized_workflow import OptimizedWorkflow

from src.api.schemas import ChatRequest, ChatResponse, SetupRequest
from pydantic import ValidationError

workflow_instances: Dict[str, OptimizedWorkflow] = {}

logger = setup_logger()

workflow = None
workflow_config = None

app = FastAPI(
    title="Banking Chatbot API",
    description="Optimized Banking Chatbot with Multi-Agent RAG",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/config")
async def setup_chat(req: SetupRequest, force: bool = False):
    global workflow, workflow_config

    if workflow and not force:
        return {
            "status": "already_initialized",
            "message": "Workflow has already been set up.",
            "current_config": workflow_config
        }

    try:
        workflow = OptimizedWorkflow(
            api_key=req.api_key,
            model=req.model,
            opensearch_endpoint=req.opensearch_endpoint,
            opensearch_username=req.opensearch_username,
            opensearch_password=req.opensearch_password
        )

        workflow_config = req.model_dump()

        return {
            "status": "success",
            "message": "Workflow initialized successfully.",
            "current_config": workflow_config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize workflow: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global workflow
    try:
        result = await workflow.run(request.question, request.previous_conversation)
        
        if hasattr(result, 'analysis') and result.analysis:
            analysis = result.analysis
            main_topic = getattr(analysis, 'main_topic', '')
            key_information = getattr(analysis, 'key_information', [])
            clarified_query = getattr(analysis, 'clarified_query', '')
            customer_type = getattr(analysis, 'customer_type', '')
        else:
            main_topic = ''
            key_information = []
            clarified_query = ''
            customer_type = ''
        
        response = ChatResponse(
            question=result.query,
            answer=result.output,
            main_topic=main_topic,
            key_information=key_information,
            clarified_query=clarified_query,
            customer_type=customer_type,
            type_of_query=getattr(result, 'type_of_query', 'banking'),
            need_human=getattr(result, 'need_human', False),
            confidence_score=1.0,
            cached=getattr(result, 'cached', False)
        )
        
        return response
        
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    global workflow, workflow_config

    if not workflow:
        return {
            "status": "not_initialized",
            "message": "No active workflow",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
        }
    
    return {
        "status": "running",
        "message": "Workflow is active",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "current_config": workflow_config,
        "workflow_class": workflow.__class__.__name__,
    }