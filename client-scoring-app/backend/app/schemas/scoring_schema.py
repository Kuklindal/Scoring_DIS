from pydantic import BaseModel
from typing import List, Optional, Dict

class ClientScore(BaseModel):
    client_id: int
    company_name: str
    code_to: Optional[int] = None
    life_span: Optional[float] = None
    number_os: Optional[float] = None
    summa_scheta_io: Optional[float] = None
    discount: Optional[float] = None
    type_of_activity: Optional[str] = None
    annual_revenue: Optional[float] = None
    number_of_employees: Optional[float] = None
    competitor: Optional[str] = None
    has_competitor: bool
    has_threat: bool
    total_session_minutes: Optional[float] = None
    final_probability: float
    risk_level: str
    feature_completeness: float
    top_factors: List[str]

class ScoreSummary(BaseModel):
    total_clients: int
    high_risk: int
    avg_risk: float
    risk_distribution: Dict[str, int]

class ScoreResponse(BaseModel):
    summary: ScoreSummary
    clients: List[ClientScore]

class FileUploadResponse(BaseModel):
    file_id: str
    message: str
    status: str