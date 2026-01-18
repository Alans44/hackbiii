
import os
import requests
import json
from typing import Optional
import base64

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', 'your-elevenlabs-api-key')
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Pre-configured voice IDs (ElevenLabs free tier includes several voices)
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Warm, friendly female voice
    "josh": "TxGEqnHWrfWFTfGW9XjX",     # Professional male voice
    "bella": "EXAVITQu4vr4xnSDxMaL",    # Soft, nurturing female voice
    "adam": "pNInz6obpgDQGcFmaJgB",     # Deep, authoritative male voice
}

# Default voice for EcoShelf
DEFAULT_VOICE = "rachel"


class ElevenLabsVoice:
    """
    ElevenLabs Text-to-Speech client for EcoShelf.
    Provides natural voice alerts for food freshness notifications.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def generate_alert(
        self, 
        text: str, 
        voice: str = DEFAULT_VOICE,
        emotion: str = "friendly"
    ) -> Optional[bytes]:
        """
        Generate voice alert for food freshness notification.
        
        Args:
            text: The message to speak
            voice: Voice ID or name from VOICES dict
            emotion: Emotional tone (friendly, urgent, calm)
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        voice_id = VOICES.get(voice, voice)
        
        # Adjust voice settings based on emotion
        voice_settings = self._get_voice_settings(emotion)
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": voice_settings
        }
        
        try:
            response = requests.post(
                f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"ElevenLabs request failed: {e}")
            return None
    
    def _get_voice_settings(self, emotion: str) -> dict:
        """Get voice settings based on emotional tone"""
        settings = {
            "friendly": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True
            },
            "urgent": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.6,
                "use_speaker_boost": True
            },
            "calm": {
                "stability": 0.7,
                "similarity_boost": 0.6,
                "style": 0.2,
                "use_speaker_boost": False
            }
        }
        return settings.get(emotion, settings["friendly"])
    
    def generate_freshness_alert(self, item: str, freshness: int) -> Optional[bytes]:
        """
        Generate contextual voice alert based on freshness level.
        
        Args:
            item: Food item name
            freshness: Freshness percentage (0-100)
            
        Returns:
            Audio data as bytes
        """
        if freshness >= 70:
            text = f"Your {item} is looking fresh! No rush to use it."
            emotion = "calm"
        elif freshness >= 40:
            text = f"Heads up! Your {item} is at {freshness}% freshness. Consider using it in the next few days."
            emotion = "friendly"
        else:
            text = f"Attention! Your {item} is only {freshness}% fresh. Use it today to avoid waste!"
            emotion = "urgent"
        
        return self.generate_alert(text, emotion=emotion)
    
    def generate_daily_summary(self, items: list) -> Optional[bytes]:
        """
        Generate daily fridge summary voice message.
        
        Args:
            items: List of dicts with 'name' and 'freshness' keys
            
        Returns:
            Audio data as bytes
        """
        urgent_items = [i for i in items if i['freshness'] < 40]
        moderate_items = [i for i in items if 40 <= i['freshness'] < 70]
        
        if urgent_items:
            urgent_names = ", ".join([i['name'] for i in urgent_items[:3]])
            text = f"Good morning! You have {len(urgent_items)} items that need attention today: {urgent_names}. "
        else:
            text = "Good morning! All your food is looking great today. "
        
        if moderate_items:
            text += f"There are {len(moderate_items)} items to use this week."
        
        text += " Have a wonderful, waste-free day!"
        
        return self.generate_alert(text, emotion="friendly")
    
    def list_available_voices(self) -> list:
        """Get list of available voices from ElevenLabs"""
        try:
            response = requests.get(
                f"{ELEVENLABS_BASE_URL}/voices",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json().get('voices', [])
        except:
            pass
        return []
    
    def get_usage_stats(self) -> dict:
        """Get current API usage statistics"""
        try:
            response = requests.get(
                f"{ELEVENLABS_BASE_URL}/user/subscription",
                headers=self.headers
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "character_count": data.get("character_count", 0),
                    "character_limit": data.get("character_limit", 10000),
                    "tier": data.get("tier", "free")
                }
        except:
            pass
        return {"error": "Unable to fetch usage stats"}


# Pre-built alert messages for common scenarios
ALERT_MESSAGES = {
    "banana_urgent": "Your bananas are getting quite ripe! Perfect for banana bread today.",
    "apple_moderate": "Your apples are still good but getting softer. Great for applesauce!",
    "lettuce_urgent": "Your lettuce needs to be used today. How about a big salad for lunch?",
    "milk_expiring": "Your milk is expiring soon. Time to finish that cereal!",
    "bread_moderate": "Your bread is a few days old. Toast it up or make some French toast!",
    "general_reminder": "Remember to check your fridge. Waste less, eat smarter with EcoShelf!",
}


# JavaScript integration for web app
ELEVENLABS_JS_INTEGRATION = '''
<!-- ElevenLabs Voice Integration for EcoShelf -->
<script>
    class EcoShelfVoice {
        constructor() {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.enabled = true;
        }
        
        async playAlert(audioBase64) {
            if (!this.enabled) return;
            
            const audioData = atob(audioBase64);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }
            
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start(0);
        }
        
        async fetchAndPlayAlert(item, freshness) {
            try {
                const response = await fetch('/api/voice-alert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ item, freshness })
                });
                const data = await response.json();
                if (data.audio) {
                    await this.playAlert(data.audio);
                }
            } catch (error) {
                console.log('Voice alert unavailable:', error);
            }
        }
        
        toggle() {
            this.enabled = !this.enabled;
            return this.enabled;
        }
    }
    
    // Initialize voice system
    const ecoShelfVoice = new EcoShelfVoice();
</script>

<button onclick="ecoShelfVoice.toggle()" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    padding: 15px;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    z-index: 1000;
" title="Toggle Voice Alerts">
    🔊
</button>
'''


if __name__ == "__main__":
    print("=" * 60)
    print("ElevenLabs Voice Integration for EcoShelf")
    print("=" * 60)
    print("\nSetup Instructions:")
    print("1. Sign up at https://elevenlabs.io (free tier available)")
    print("2. Get your API key from Settings")
    print("3. Set ELEVENLABS_API_KEY environment variable")
    print("\nAvailable Voices:", list(VOICES.keys()))
    print("\nExample Alert Messages:")
    for key, msg in ALERT_MESSAGES.items():
        print(f"  {key}: {msg[:50]}...")
