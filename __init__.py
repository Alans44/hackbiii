from .auth0_config import (
    AUTH0_DOMAIN,
    AUTH0_CLIENT_ID,
    requires_auth,
    requires_scope,
    get_auth0_login_url,
    AUTH0_LOGIN_TEMPLATE
)

from .digitalocean_gradient import (
    DigitalOceanGradient,
    GRADIENT_MODEL_CONFIG
)

from .elevenlabs_voice import (
    ElevenLabsVoice,
    ALERT_MESSAGES,
    ELEVENLABS_JS_INTEGRATION
)

from .genai_recipes import (
    EcoShelfRecipeEngine,
    OpenAIRecipes,
    AnthropicRecipes,
    GeminiRecipes,
    calculate_recipe_score
)

from .mongodb_atlas import (
    MongoDBAtlas,
    SCHEMAS,
    INDEXES
)

from .solana_tracker import (
    SolanaWasteTracker
)

__all__ = [
    # Auth0
    'AUTH0_DOMAIN',
    'AUTH0_CLIENT_ID', 
    'requires_auth',
    'requires_scope',
    'get_auth0_login_url',
    'AUTH0_LOGIN_TEMPLATE',
    
    # DigitalOcean
    'DigitalOceanGradient',
    'GRADIENT_MODEL_CONFIG',
    
    # ElevenLabs
    'ElevenLabsVoice',
    'ALERT_MESSAGES',
    'ELEVENLABS_JS_INTEGRATION',
    
    # GenAI
    'EcoShelfRecipeEngine',
    'OpenAIRecipes',
    'AnthropicRecipes',
    'GeminiRecipes',
    'calculate_recipe_score',
    
    # MongoDB
    'MongoDBAtlas',
    'SCHEMAS',
    'INDEXES',
    
    # Solana
    'SolanaWasteTracker',
]

# Version
__version__ = '1.0.0'
