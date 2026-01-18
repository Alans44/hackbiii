import os
from functools import wraps
from flask import request, jsonify, session, redirect, url_for
import json
from urllib.request import urlopen
from jose import jwt

# Auth0 Configuration
AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN', 'your-tenant.auth0.com')
AUTH0_CLIENT_ID = os.environ.get('AUTH0_CLIENT_ID', 'your-client-id')
AUTH0_CLIENT_SECRET = os.environ.get('AUTH0_CLIENT_SECRET', 'your-client-secret')
AUTH0_CALLBACK_URL = os.environ.get('AUTH0_CALLBACK_URL', 'http://localhost:5000/callback')
AUTH0_AUDIENCE = os.environ.get('AUTH0_AUDIENCE', f'https://{AUTH0_DOMAIN}/api/v2/')

# Algorithm used by Auth0
ALGORITHMS = ["RS256"]


class Auth0Error(Exception):
    """Auth0 specific error"""
    pass


def get_auth0_public_key():
    """Fetch Auth0 public key for JWT verification"""
    jsonurl = urlopen(f"https://{AUTH0_DOMAIN}/.well-known/jwks.json")
    jwks = json.loads(jsonurl.read())
    return jwks


def get_token_auth_header():
    """Extract the Access Token from the Authorization Header"""
    auth = request.headers.get("Authorization", None)
    if not auth:
        raise Auth0Error("Authorization header is expected")

    parts = auth.split()

    if parts[0].lower() != "bearer":
        raise Auth0Error("Authorization header must start with Bearer")
    elif len(parts) == 1:
        raise Auth0Error("Token not found")
    elif len(parts) > 2:
        raise Auth0Error("Authorization header must be Bearer token")

    token = parts[1]
    return token


def requires_auth(f):
    """
    Decorator to protect routes with Auth0 authentication.
    Use this on any endpoint that requires a logged-in user.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = get_token_auth_header()
            jwks = get_auth0_public_key()
            
            unverified_header = jwt.get_unverified_header(token)
            
            rsa_key = {}
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"]
                    }
            
            if rsa_key:
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=ALGORITHMS,
                    audience=AUTH0_AUDIENCE,
                    issuer=f"https://{AUTH0_DOMAIN}/"
                )
                
                # Add user info to request context
                request.user = payload
                return f(*args, **kwargs)
            
            raise Auth0Error("Unable to find appropriate key")
            
        except jwt.ExpiredSignatureError:
            raise Auth0Error("Token is expired")
        except jwt.JWTClaimsError:
            raise Auth0Error("Invalid claims")
        except Exception as e:
            raise Auth0Error(f"Unable to parse authentication token: {str(e)}")
    
    return decorated


def requires_scope(required_scope):
    """
    Decorator to check if the user has the required scope.
    Use for fine-grained permission control.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = get_token_auth_header()
            unverified_claims = jwt.get_unverified_claims(token)
            
            if unverified_claims.get("scope"):
                token_scopes = unverified_claims["scope"].split()
                for scope in token_scopes:
                    if scope == required_scope:
                        return f(*args, **kwargs)
            
            return jsonify({"error": "Insufficient permissions"}), 403
        return decorated
    return decorator


# Auth0 login URL generator
def get_auth0_login_url(state=None):
    """Generate Auth0 Universal Login URL"""
    params = {
        'client_id': AUTH0_CLIENT_ID,
        'redirect_uri': AUTH0_CALLBACK_URL,
        'response_type': 'code',
        'scope': 'openid profile email',
        'audience': AUTH0_AUDIENCE,
    }
    if state:
        params['state'] = state
    
    query = '&'.join(f"{k}={v}" for k, v in params.items())
    return f"https://{AUTH0_DOMAIN}/authorize?{query}"


# HTML template for Auth0 login button
AUTH0_LOGIN_TEMPLATE = '''
<!-- Auth0 Login Integration -->
<script src="https://cdn.auth0.com/js/auth0-spa-js/2.0/auth0-spa-js.production.js"></script>
<script>
    let auth0Client = null;

    const configureAuth0 = async () => {
        auth0Client = await auth0.createAuth0Client({
            domain: "''' + AUTH0_DOMAIN + '''",
            clientId: "''' + AUTH0_CLIENT_ID + '''",
            authorizationParams: {
                redirect_uri: window.location.origin,
                audience: "''' + AUTH0_AUDIENCE + '''"
            }
        });
        
        // Check if returning from Auth0 login
        if (window.location.search.includes("code=")) {
            await auth0Client.handleRedirectCallback();
            window.history.replaceState({}, document.title, window.location.pathname);
        }
        
        updateUI();
    };

    const login = async () => {
        await auth0Client.loginWithRedirect();
    };

    const logout = () => {
        auth0Client.logout({
            logoutParams: {
                returnTo: window.location.origin
            }
        });
    };

    const updateUI = async () => {
        const isAuthenticated = await auth0Client.isAuthenticated();
        const loginBtn = document.getElementById('auth0-login');
        const logoutBtn = document.getElementById('auth0-logout');
        const userInfo = document.getElementById('auth0-user');
        
        if (isAuthenticated) {
            const user = await auth0Client.getUser();
            if (loginBtn) loginBtn.style.display = 'none';
            if (logoutBtn) logoutBtn.style.display = 'block';
            if (userInfo) userInfo.textContent = `Welcome, ${user.name}!`;
        } else {
            if (loginBtn) loginBtn.style.display = 'block';
            if (logoutBtn) logoutBtn.style.display = 'none';
            if (userInfo) userInfo.textContent = '';
        }
    };

    // Initialize on page load
    window.onload = configureAuth0;
</script>

<div id="auth0-container" style="position: absolute; top: 20px; right: 20px; z-index: 1000;">
    <span id="auth0-user" style="color: white; margin-right: 10px;"></span>
    <button id="auth0-login" onclick="login()" style="
        background: linear-gradient(135deg, #635bff, #00d4ff);
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        font-weight: 600;
    ">🔐 Sign In with Auth0</button>
    <button id="auth0-logout" onclick="logout()" style="
        display: none;
        background: #ff4757;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        font-weight: 600;
    ">Sign Out</button>
</div>
'''

if __name__ == "__main__":
    print("Auth0 Configuration:")
    print(f"  Domain: {AUTH0_DOMAIN}")
    print(f"  Client ID: {AUTH0_CLIENT_ID}")
    print(f"  Callback URL: {AUTH0_CALLBACK_URL}")
    print("\nTo set up Auth0:")
    print("1. Create a free account at https://auth0.com")
    print("2. Create a new Application (Single Page Web Application)")
    print("3. Set environment variables: AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET")
    print("4. Add callback URL to Allowed Callback URLs in Auth0 dashboard")
