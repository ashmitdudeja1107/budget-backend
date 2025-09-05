from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import firebase_admin
from firebase_admin import auth, credentials
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            # Debug: Check if .env file exists
            env_file_path = ".env"
            if os.path.exists(env_file_path):
                logger.info(f".env file found at: {os.path.abspath(env_file_path)}")
            else:
                logger.warning(f".env file not found at: {os.path.abspath(env_file_path)}")
            
            # Using environment variables to create credentials
            firebase_type = os.environ.get("FIREBASE_TYPE")
            firebase_project_id = os.environ.get("FIREBASE_PROJECT_ID")
            firebase_private_key_id = os.environ.get("FIREBASE_PRIVATE_KEY_ID")
            firebase_private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
            firebase_client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")
            firebase_client_id = os.environ.get("FIREBASE_CLIENT_ID")
            firebase_auth_uri = os.environ.get("FIREBASE_AUTH_URI")
            firebase_token_uri = os.environ.get("FIREBASE_TOKEN_URI")
            firebase_auth_provider_cert_url = os.environ.get("FIREBASE_AUTH_PROVIDER_CERT_URL")
            firebase_client_cert_url = os.environ.get("FIREBASE_CLIENT_CERT_URL")
            
            # Debug: Log which variables are missing
            required_vars = {
                "FIREBASE_TYPE": firebase_type,
                "FIREBASE_PROJECT_ID": firebase_project_id,
                "FIREBASE_PRIVATE_KEY": firebase_private_key,
                "FIREBASE_CLIENT_EMAIL": firebase_client_email
            }
            
            missing_vars = [var_name for var_name, var_value in required_vars.items() if not var_value]
            if missing_vars:
                logger.error(f"Missing environment variables: {missing_vars}")
                logger.info("Available environment variables starting with 'FIREBASE_':")
                firebase_vars = {k: v[:50] + "..." if v and len(v) > 50 else v 
                               for k, v in os.environ.items() if k.startswith("FIREBASE_")}
                for var, value in firebase_vars.items():
                    logger.info(f"  {var}: {value}")
            
            if all([firebase_type, firebase_project_id, firebase_private_key, firebase_client_email]):
                # Create credentials dict from environment variables
                cred_dict = {
                    "type": firebase_type,
                    "project_id": firebase_project_id,
                    "private_key_id": firebase_private_key_id,
                    "private_key": firebase_private_key.replace('\\n', '\n'),  # Handle newlines properly
                    "client_email": firebase_client_email,
                    "client_id": firebase_client_id,
                    "auth_uri": firebase_auth_uri,
                    "token_uri": firebase_token_uri,
                    "auth_provider_x509_cert_url": firebase_auth_provider_cert_url,
                    "client_x509_cert_url": firebase_client_cert_url
                }
                
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully with environment variables")
            else:
                logger.error(f"Missing required Firebase environment variables: {missing_vars}")
                raise Exception(f"Missing required Firebase environment variables: {missing_vars}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise e

# Initialize Firebase when module is loaded
initialize_firebase()

router = APIRouter(prefix="/auth", tags=["authentication"])

# Pydantic models for request/response
class GoogleAuthRequest(BaseModel):
    token: str

class UserResponse(BaseModel):
    user_id: int
    name: str
    email: str

class GoogleSignupResponse(BaseModel):
    message: str
    user: UserResponse

class GoogleLoginResponse(BaseModel):
    message: str
    token: str
    user: UserResponse

# GOOGLE SIGNUP ROUTE
@router.post("/google-signup", response_model=GoogleSignupResponse, status_code=status.HTTP_201_CREATED)
async def google_signup(request: GoogleAuthRequest):
    try:
        logger.info(f"Google signup request received: {request.dict()}")
        
        if not request.token:
            logger.info("No token provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token is required"
            )

        logger.info("Verifying Firebase token...")
        try:
            decoded = auth.verify_id_token(request.token)
            logger.info(f"Token decoded successfully: email={decoded.get('email')}, name={decoded.get('name', decoded.get('displayName'))}")
        except Exception as token_error:
            logger.error(f"Token verification failed: {token_error}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {str(token_error)}"
            )
        
        email = decoded.get('email')
        name = decoded.get('name') or decoded.get('displayName') or "Unnamed User"

        logger.info(f"Creating new user: name={name}, email={email}")
        
        # Mock user creation - no database operations
        mock_user_id = 12345
        logger.info(f"User created successfully with ID: {mock_user_id}")
        
        # Return user data
        return GoogleSignupResponse(
            message="Google signup successful",
            user=UserResponse(
                user_id=mock_user_id,
                name=name,
                email=email
            )
        )
    
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as error:
        logger.error(f"Google signup error details: message={str(error)}, type={type(error).__name__}")
        
        # More specific error handling for Firebase errors
        error_code = getattr(error, 'code', None)
        if error_code == 'auth/id-token-expired':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        elif error_code == 'auth/invalid-id-token':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        elif error_code == 'auth/id-token-revoked':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token revoked"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(error)}"
        )


# GOOGLE LOGIN ROUTE
@router.post("/google-login", response_model=GoogleLoginResponse)
async def google_login(request: GoogleAuthRequest):
    try:
        logger.info(f"Google login request received: {request.dict()}")
        
        if not request.token:
            logger.info("No token provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token is required"
            )

        logger.info("Verifying Firebase token...")
        try:
            decoded = auth.verify_id_token(request.token)
            logger.info(f"Token decoded successfully: email={decoded.get('email')}, name={decoded.get('name', decoded.get('displayName'))}")
        except Exception as token_error:
            logger.error(f"Token verification failed: {token_error}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {str(token_error)}"
            )
        
        email = decoded.get('email')
        name = decoded.get('name') or decoded.get('displayName') or "Unnamed User"

        logger.info("Google login successful")
        
        # Return user data (mock response for now)
        return GoogleLoginResponse(
            message="Google login successful",
            token=request.token,
            user=UserResponse(
                user_id=12345,  # Mock user ID
                name=name,
                email=email
            )
        )
    
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as error:
        logger.error(f"Google login error details: message={str(error)}, type={type(error).__name__}")
        
        # More specific error handling for Firebase errors
        error_code = getattr(error, 'code', None)
        if error_code == 'auth/id-token-expired':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        elif error_code == 'auth/invalid-id-token':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        elif error_code == 'auth/id-token-revoked':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token revoked"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(error)}"
        )