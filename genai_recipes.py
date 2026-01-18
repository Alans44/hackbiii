"""
Generative AI Recipe Integration for EcoShelf
===============================================
AI-powered recipe suggestions using multiple GenAI providers.

MLH Prize: Best Use of Gen AI

Supports:
- OpenAI GPT-4
- Anthropic Claude
- Google Gemini
- Hugging Face models
"""

import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# API Keys from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')


class GenAIProvider(ABC):
    """Abstract base class for GenAI providers"""
    
    @abstractmethod
    def generate_recipes(self, ingredients: List[str], urgent_items: List[str]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_storage_tips(self, food_item: str, freshness: int) -> str:
        pass


class OpenAIRecipes(GenAIProvider):
    """OpenAI GPT-4 recipe generator"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = "gpt-4-turbo-preview"
    
    def generate_recipes(self, ingredients: List[str], urgent_items: List[str] = None) -> Dict[str, Any]:
        """Generate recipes using OpenAI GPT-4"""
        import requests
        
        prompt = self._build_recipe_prompt(ingredients, urgent_items)
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful chef focused on reducing food waste. Always prioritize using ingredients that are about to expire."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return {"recipes": content, "provider": "OpenAI GPT-4"}
        return {"error": response.text}
    
    def get_storage_tips(self, food_item: str, freshness: int) -> str:
        """Get storage tips using GPT-4"""
        import requests
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a food storage expert. Give concise, practical advice."},
                    {"role": "user", "content": f"My {food_item} is at {freshness}% freshness. How should I store it and what can I make with it?"}
                ],
                "max_tokens": 300
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "Unable to get storage tips."
    
    def _build_recipe_prompt(self, ingredients: List[str], urgent_items: List[str] = None) -> str:
        prompt = f"I have these ingredients: {', '.join(ingredients)}.\n"
        if urgent_items:
            prompt += f"These items need to be used TODAY: {', '.join(urgent_items)}.\n"
        prompt += """
Please suggest 3 recipes that:
1. MUST use the urgent items first
2. Are quick to make (under 30 minutes)
3. Minimize food waste

For each recipe provide:
- Name
- Time to prepare
- Ingredients used from my list
- Brief instructions (3-4 steps)
- Why this helps reduce waste

Format as a numbered list."""
        return prompt


class AnthropicRecipes(GenAIProvider):
    """Anthropic Claude recipe generator"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = "claude-3-sonnet-20240229"
    
    def generate_recipes(self, ingredients: List[str], urgent_items: List[str] = None) -> Dict[str, Any]:
        """Generate recipes using Claude"""
        import requests
        
        prompt = f"""I have these ingredients in my fridge: {', '.join(ingredients)}.
{'Items that MUST be used today: ' + ', '.join(urgent_items) if urgent_items else ''}

Suggest 3 waste-reducing recipes. For each, include:
- Recipe name
- Prep time
- Which of my ingredients it uses
- Quick instructions
- Waste reduction tip"""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": self.model,
                "max_tokens": 1500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        if response.status_code == 200:
            content = response.json()['content'][0]['text']
            return {"recipes": content, "provider": "Anthropic Claude"}
        return {"error": response.text}
    
    def get_storage_tips(self, food_item: str, freshness: int) -> str:
        """Get storage tips using Claude"""
        import requests
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": self.model,
                "max_tokens": 300,
                "messages": [
                    {"role": "user", "content": f"My {food_item} is {freshness}% fresh. Quick storage tips and recipe ideas?"}
                ]
            }
        )
        
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        return "Unable to get storage tips."


class GeminiRecipes(GenAIProvider):
    """Google Gemini recipe generator"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model = "gemini-pro"
    
    def generate_recipes(self, ingredients: List[str], urgent_items: List[str] = None) -> Dict[str, Any]:
        """Generate recipes using Gemini"""
        import requests
        
        prompt = f"""You are a chef helping reduce food waste.
        
Available ingredients: {', '.join(ingredients)}
{'URGENT - use today: ' + ', '.join(urgent_items) if urgent_items else ''}

Give me 3 quick recipes (under 30 min) that prioritize the urgent items.
Include: name, time, ingredients used, steps, waste-saving tip."""

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1500
                }
            }
        )
        
        if response.status_code == 200:
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            return {"recipes": content, "provider": "Google Gemini"}
        return {"error": response.text}
    
    def get_storage_tips(self, food_item: str, freshness: int) -> str:
        """Get storage tips using Gemini"""
        import requests
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": f"Storage tips for {food_item} at {freshness}% freshness?"}]}]
            }
        )
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return "Unable to get storage tips."


class EcoShelfRecipeEngine:
    """
    Unified recipe engine that uses multiple GenAI providers.
    Automatically falls back if one provider fails.
    """
    
    def __init__(self):
        self.providers = []
        
        # Initialize available providers based on API keys
        if OPENAI_API_KEY:
            self.providers.append(OpenAIRecipes())
        if ANTHROPIC_API_KEY:
            self.providers.append(AnthropicRecipes())
        if GOOGLE_API_KEY:
            self.providers.append(GeminiRecipes())
    
    def get_recipes(self, ingredients: List[str], urgent_items: List[str] = None) -> Dict[str, Any]:
        """
        Generate recipes using available GenAI providers.
        Falls back to next provider if one fails.
        """
        for provider in self.providers:
            try:
                result = provider.generate_recipes(ingredients, urgent_items)
                if "error" not in result:
                    return result
            except Exception as e:
                continue
        
        # Fallback: return basic suggestions without AI
        return self._fallback_recipes(ingredients, urgent_items)
    
    def get_tips(self, food_item: str, freshness: int) -> str:
        """Get storage tips from available provider"""
        for provider in self.providers:
            try:
                tips = provider.get_storage_tips(food_item, freshness)
                if tips and "Unable" not in tips:
                    return tips
            except:
                continue
        
        return self._fallback_tips(food_item, freshness)
    
    def _fallback_recipes(self, ingredients: List[str], urgent_items: List[str] = None) -> Dict[str, Any]:
        """Basic recipe suggestions when AI is unavailable"""
        recipes = []
        
        if 'banana' in ingredients:
            recipes.append({
                "name": "Quick Banana Smoothie",
                "time": "5 minutes",
                "ingredients": ["banana", "milk (or water)", "honey"],
                "steps": "Blend banana with liquid until smooth. Add sweetener to taste."
            })
        
        if 'apple' in ingredients:
            recipes.append({
                "name": "Simple Apple Slices with Peanut Butter",
                "time": "3 minutes",
                "ingredients": ["apple", "peanut butter"],
                "steps": "Slice apple and serve with peanut butter for dipping."
            })
        
        if any(item in ingredients for item in ['broccoli', 'carrot']):
            recipes.append({
                "name": "Quick Veggie Stir Fry",
                "time": "15 minutes",
                "ingredients": ["vegetables", "oil", "soy sauce"],
                "steps": "Chop veggies, stir fry in oil, season with soy sauce."
            })
        
        return {
            "recipes": recipes,
            "provider": "EcoShelf Fallback",
            "note": "Connect a GenAI API for personalized recipes!"
        }
    
    def _fallback_tips(self, food_item: str, freshness: int) -> str:
        """Basic storage tips when AI is unavailable"""
        tips = {
            "banana": "Store at room temperature. Ripe bananas can be frozen for smoothies.",
            "apple": "Keep in the fridge crisper drawer. Apples release ethylene gas.",
            "orange": "Store at room temperature or refrigerate for longer life.",
            "broccoli": "Keep in plastic bag in fridge. Use within 3-5 days.",
            "carrot": "Remove greens and store in water in the fridge.",
        }
        return tips.get(food_item.lower(), f"Store {food_item} properly to maximize freshness.")


# Recipe scoring algorithm for waste reduction
def calculate_recipe_score(recipe_ingredients: List[str], 
                          available_items: Dict[str, int],
                          weights: Dict[str, float] = None) -> float:
    """
    Score a recipe based on waste reduction potential.
    
    Score = α * Overlap + β * Σ Urgency - γ * Missing
    
    Args:
        recipe_ingredients: List of ingredients needed
        available_items: Dict of {item: freshness_score}
        weights: Dict with 'alpha', 'beta', 'gamma' weights
    
    Returns:
        Recipe score (higher = better for waste reduction)
    """
    if weights is None:
        weights = {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.3}
    
    available_names = set(available_items.keys())
    recipe_set = set(recipe_ingredients)
    
    # Overlap: how many recipe ingredients we have
    overlap = len(recipe_set & available_names) / len(recipe_set)
    
    # Urgency: sum of urgency scores for ingredients we'd use
    urgency_sum = 0
    for item in recipe_set & available_names:
        freshness = available_items[item]
        # Urgency inversely proportional to freshness
        urgency_sum += max(0, 1 / (freshness / 10 + 1))
    
    # Missing: penalty for ingredients we don't have
    missing = len(recipe_set - available_names)
    
    score = (weights['alpha'] * overlap + 
             weights['beta'] * urgency_sum - 
             weights['gamma'] * missing)
    
    return round(score, 2)


if __name__ == "__main__":
    print("=" * 60)
    print("EcoShelf Generative AI Recipe Engine")
    print("=" * 60)
    
    print("\nSupported Providers:")
    print("  - OpenAI GPT-4 (set OPENAI_API_KEY)")
    print("  - Anthropic Claude (set ANTHROPIC_API_KEY)")
    print("  - Google Gemini (set GOOGLE_API_KEY)")
    
    print("\nExample Usage:")
    print("""
    engine = EcoShelfRecipeEngine()
    recipes = engine.get_recipes(
        ingredients=['banana', 'apple', 'milk', 'bread'],
        urgent_items=['banana']  # Use today!
    )
    print(recipes)
    """)
    
    # Demo with fallback
    engine = EcoShelfRecipeEngine()
    result = engine.get_recipes(['banana', 'apple', 'broccoli'])
    print("\nSample Output (Fallback Mode):")
    print(json.dumps(result, indent=2))
