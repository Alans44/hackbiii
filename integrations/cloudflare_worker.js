/**
 * Cloudflare Workers AI Integration for EcoShelf
 * ================================================
 * Uses Cloudflare Workers AI for serverless freshness analysis.
 * 
 * MLH Prize: Best AI Application Built with Cloudflare
 * 
 * Deploy with: wrangler deploy
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // CORS headers for cross-origin requests
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Route handling
    if (url.pathname === '/api/analyze-freshness') {
      return handleFreshnessAnalysis(request, env, corsHeaders);
    }
    
    if (url.pathname === '/api/generate-recipe') {
      return handleRecipeGeneration(request, env, corsHeaders);
    }
    
    if (url.pathname === '/api/food-tips') {
      return handleFoodTips(request, env, corsHeaders);
    }

    // Health check
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({ 
        status: 'healthy',
        service: 'EcoShelf Cloudflare Worker',
        ai_enabled: true
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      message: 'EcoShelf Cloudflare Workers AI',
      endpoints: [
        '/api/analyze-freshness - POST image for AI freshness analysis',
        '/api/generate-recipe - POST ingredients for recipe suggestions',
        '/api/food-tips - POST food item for storage tips',
        '/health - Service health check'
      ]
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  },
};

/**
 * Analyze food freshness using Cloudflare Workers AI (LLaVA vision model)
 */
async function handleFreshnessAnalysis(request, env, corsHeaders) {
  try {
    const formData = await request.formData();
    const image = formData.get('image');
    
    if (!image) {
      return new Response(JSON.stringify({ error: 'No image provided' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Convert image to array buffer for AI model
    const imageData = await image.arrayBuffer();
    const imageArray = [...new Uint8Array(imageData)];

    // Use Cloudflare Workers AI - LLaVA model for vision
    const response = await env.AI.run('@cf/llava-hf/llava-1.5-7b-hf', {
      image: imageArray,
      prompt: `Analyze this food image and provide:
1. What food items are visible
2. Estimated freshness level (0-100%)
3. Signs of spoilage or ripeness
4. Recommended days until consumption
5. Storage tips

Format as JSON with keys: items, freshness_percent, observations, days_remaining, tips`,
      max_tokens: 500
    });

    return new Response(JSON.stringify({
      success: true,
      analysis: response,
      model: 'llava-1.5-7b-hf',
      powered_by: 'Cloudflare Workers AI'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    return new Response(JSON.stringify({ 
      error: error.message,
      hint: 'Ensure AI binding is configured in wrangler.toml'
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Generate recipes based on available ingredients using Cloudflare Workers AI
 */
async function handleRecipeGeneration(request, env, corsHeaders) {
  try {
    const { ingredients, urgentItems } = await request.json();
    
    if (!ingredients || ingredients.length === 0) {
      return new Response(JSON.stringify({ error: 'No ingredients provided' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Use Cloudflare Workers AI - Llama model for text generation
    const prompt = `You are a helpful chef. Given these ingredients: ${ingredients.join(', ')}.
${urgentItems ? `These items need to be used soon: ${urgentItems.join(', ')}.` : ''}

Suggest 3 simple recipes that:
1. Prioritize using the urgent items first
2. Minimize food waste
3. Are easy to make (under 30 minutes)

For each recipe provide: name, ingredients used, brief instructions, and prep time.
Format as JSON array.`;

    const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
      messages: [
        { role: 'system', content: 'You are a helpful cooking assistant focused on reducing food waste.' },
        { role: 'user', content: prompt }
      ],
      max_tokens: 1000
    });

    return new Response(JSON.stringify({
      success: true,
      recipes: response,
      model: 'llama-3-8b-instruct',
      powered_by: 'Cloudflare Workers AI'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get food storage tips using Cloudflare Workers AI
 */
async function handleFoodTips(request, env, corsHeaders) {
  try {
    const { foodItem, currentFreshness } = await request.json();
    
    const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
      messages: [
        { role: 'system', content: 'You are a food storage expert helping reduce household food waste.' },
        { role: 'user', content: `For ${foodItem} at ${currentFreshness}% freshness, provide:
1. Best storage method
2. Signs it's going bad
3. How to extend shelf life
4. Creative uses if it's getting old
Keep response concise and practical.` }
      ],
      max_tokens: 300
    });

    return new Response(JSON.stringify({
      success: true,
      tips: response,
      foodItem,
      powered_by: 'Cloudflare Workers AI'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}
