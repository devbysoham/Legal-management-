import google.generativeai as genai
from google.generativeai.types.content_types import to_schema
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Test 1: Model with Field(None, ...) which generates "default": null
class ComplianceCheckWithDefault(BaseModel):
    regulation: str
    recommendation: Optional[str] = Field(None, description="Recommendation")

try:
    print("Testing ComplianceCheckWithDefault...")
    schema = to_schema(ComplianceCheckWithDefault)
    print("Successfully generated schema for default!")
except Exception as e:
    print("Failed as expected:", e)

# Test 2: Model without default value in Field
class ComplianceCheckNoDefault(BaseModel):
    regulation: str
    recommendation: Optional[str] = Field(description="Recommendation")

try:
    print("\nTesting ComplianceCheckNoDefault...")
    schema = to_schema(ComplianceCheckNoDefault)
    print("Successfully generated schema for no default!")
except Exception as e:
    print("Failed:", e)
