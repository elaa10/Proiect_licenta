from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    result = relationship("AnalysisResult", back_populates="request", uselist=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(
        Integer, ForeignKey("analysis_requests.id"), nullable=False, unique=True
    )
    lexical_score = Column(Float, nullable=True)
    ml_score = Column(Float, nullable=True)
    screenshot_path = Column(String(500), nullable=True)
    visual_match_brand = Column(String(100), nullable=True)
    visual_similarity = Column(Float, nullable=True)
    verdict = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    request = relationship("AnalysisRequest", back_populates="result")

    visual_model = Column(String(10), nullable=True) 


class BrandReference(Base):
    __tablename__ = "brand_references"

    id = Column(Integer, primary_key=True, index=True)
    brand_name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50), nullable=True)
    screenshot_path = Column(String(500), nullable=True)
    embedding_vector = Column(Text, nullable=True)