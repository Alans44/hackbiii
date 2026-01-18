"""
EcoShelf Integrations Package
==============================

This package contains integrations with various MLH sponsor technologies:

- Auth0: Secure authentication (auth0_config.py)
- Cloudflare Workers AI: Serverless AI inference (cloudflare_worker.js)
- DigitalOcean Gradient: GPU-accelerated ML (digitalocean_gradient.py)
- ElevenLabs: Voice alerts (elevenlabs_voice.py)
- Generative AI: Recipe suggestions (genai_recipes.py)
- MongoDB Atlas: Cloud database (mongodb_atlas.py)
- Solana: Blockchain tracking (solana_tracker.py)

MLH Prizes Targeted:
- Best Use of Auth0
- Best AI Application Built with Cloudflare
- Best Use of DigitalOcean Gradient™ AI
- Best Use of ElevenLabs
- Best Use of Gen AI
- Best Use of MongoDB Atlas
- Best Use of Solana
"""

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
