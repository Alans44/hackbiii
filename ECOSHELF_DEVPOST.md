# 🥬 EcoShelf: See What's Fresh, Waste Less

> **Waste less. Eat smarter.**

---

## 💡 Inspiration

Food waste is one of those problems that's both massive and strangely personal. We've all opened the fridge, stared at that banana or bag of spinach, and wondered: *"Is this still good?"* That moment of uncertainty leads to two outcomes—either we toss perfectly fine food "just to be safe," or we forget about it entirely until it's definitely not fine.

That hesitation adds up. In the US alone, **40% of food goes to waste**, with fresh produce being the biggest culprit. But here's the thing: most of that waste isn't because food actually went bad—it's because we couldn't tell when it was about to.

We built **EcoShelf** to remove that guesswork. Point a camera at your fridge, see exactly what you have, and know at a glance what needs to be used soon. No apps to update manually. No scanning barcodes. Just look and know.

---

## 🍎 What It Does

EcoShelf is a **real-time food freshness monitor** that turns your camera into a smart fridge assistant.

### Core Features

| Feature | Description |
|---------|-------------|
| 📸 **Instant Food Detection** | Point the camera at your fridge and EcoShelf automatically identifies produce, bottles, snacks, sandwiches, and more |
| 🔬 **Freshness Analysis** | Each detected item gets a freshness score (0-100%) based on visual cues like color, texture, and ripeness |
| 🚦 **Color-Coded Status** | Items tagged as **Fresh** (green), **Moderate** (orange), or **Spoiling** (red) for at-a-glance prioritization |
| 📋 **Live Detection Panel** | Clean sidebar showing everything visible, updating in real-time as items move in/out of frame |
| 🖥️ **Beautiful Fridge UI** | Camera feed displayed inside a stylized fridge graphic—like looking into your actual fridge |

### What It Detects

| Category | Items |
|----------|-------|
| 🍎 **Produce** | Apples, bananas, oranges, broccoli, carrots |
| 🍼 **Bottles** | Water bottles, plastic bottles |
| 🍪 **Snacks** | Cakes, donuts, pastries |
| 🥪 **Proteins** | Sandwiches, hot dogs |
| 🍕 **Prepared Foods** | Pizza |

### How Freshness Works

Each produce item receives a freshness score from our trained neural network:

| Score | Status | What It Means |
|-------|--------|---------------|
| **70-100%** | 🟢 Fresh | Good to go, no rush |
| **40-69%** | 🟠 Moderate | Use within a few days |
| **0-39%** | 🔴 Spoiling | Use today or consider composting |

The prioritization follows a simple urgency model—items closer to expiring get flagged first:

$$\text{Urgency}(i) \propto \max\left(0, \frac{1}{\text{days\_to\_expire}(i) + 1}\right)$$

---

## 🛠️ How We Built It

### The Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Camera    │────▶│  YOLO Detection  │────▶│ Freshness Model │
│  (OpenCV)   │     │   (YOLOv8 Nano)  │     │   (ResNet-18)   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                    ┌──────────────────┐              ▼
                    │   Flask Server   │◀────────────────────────
                    │  (MJPEG Stream)  │     Freshness Scores
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Web Frontend   │
                    │ (Fridge UI + JS) │
                    └──────────────────┘
```

### Tech Stack

**🧠 Computer Vision**
- **YOLOv8 Nano** — Ultrafast object detection (~30+ FPS) optimized for real-time performance
- **Custom ResNet-18 Freshness Model** — Trained classifier that scores produce ripeness from visual features (color, texture, spotting, wilting)
- **OpenCV** — Camera capture, frame processing, and MJPEG video streaming

**⚙️ Backend**
- **Flask** — Lightweight Python web server
- **Flask-CORS** — Cross-origin support for web interface
- **MJPEG Streaming** — Real-time video feed to browser
- **Threading** — Non-blocking detection pipeline

**🎨 Frontend**
- **Embedded HTML/CSS/JS** — Single-file deployment, no build step required
- **Modern CSS** — Glassmorphism effects, smooth gradients, responsive animations
- **Real-time Updates** — Detection panel refreshes every 500ms via REST API

**⚡ Performance Optimizations**
- Frame skipping (process every 2nd frame)
- Reduced inference resolution (480px) while maintaining HD display (1280×720)
- Buffer size optimization to minimize camera latency
- Nano YOLO model for speed over marginal accuracy gains

---

## 🚧 Challenges We Faced

### Freshness Isn't Binary
**Challenge:** Different foods age differently, and storage conditions matter. A slightly spotted banana might be perfect for baking but not for eating fresh.

**Solution:** We designed outputs as confidence-based ranges rather than hard predictions. The scoring system communicates uncertainty honestly—users see a percentage, not a false "EXPIRED" stamp.

### Messy Fridges Are Real
**Challenge:** Items overlap, labels face away, lighting varies wildly. A model trained on perfect stock photos struggles with real-world chaos.

**Solution:** Made the experience resilient. Partial recognition is still useful—if we can only see half a banana, we still detect "banana." Users can quickly see what's visible and adjust accordingly.

### Speed vs. Accuracy Tradeoff
**Challenge:** A hackathon demo needs to feel *instant*. Heavy models give better accuracy but kill responsiveness.

**Solution:** Chose YOLOv8 Nano over larger variants. Implemented aggressive frame skipping and resolution scaling. The result: smooth real-time video with detection overlays that don't lag behind reality.

### Making It Actually Useful
**Challenge:** It's easy to detect things. It's hard to present that information in a way that changes behavior.

**Solution:** Focused on the UI. The fridge graphic isn't just decoration—it creates a mental model. The color-coded urgency system turns "83% fresh" into an instant gut reaction: *green = don't worry, red = eat today*.

---

## 🎓 What We Learned

1. **Combining vision with heuristics is as much about communicating uncertainty as prediction.** Clear outputs build trust. A confident wrong answer is worse than an honest "probably fresh."

2. **"Smart" features only matter if the flow is effortless.** We treated friction like a bug. Every extra click, every loading spinner, every confusing label—removed.

3. **The UI is the product.** Our model accuracy matters less than whether someone *glances at the screen and immediately knows what to do*.

4. **Real-time is harder than it looks.** Threading, buffers, frame rates, latency—getting smooth video with overlays required way more optimization than expected.

---

## 🚀 What's Next

| Feature | Description |
|---------|-------------|
| 📦 **More Food Types** | Packaged items, leftovers, meal prep containers |
| 📊 **Personalized Estimates** | Learn from user habits to calibrate shelf-life predictions |
| 📱 **Mobile App** | Take EcoShelf on-the-go with push notifications |
| 🛒 **Recipe Suggestions** | "You have bananas at 45% freshness—here are 5 banana bread recipes" |
| 🔒 **On-Device Inference** | Privacy-first option with no cloud dependency |
| 📋 **Inventory Tracking** | Build a persistent list of fridge contents over time |

---

## 🏗️ Built With

### Core Technology
- **YOLOv8** — Real-time object detection
- **PyTorch + ResNet-18** — Freshness classification model
- **OpenCV** — Camera capture and image processing
- **Flask** — Python web server
- **HTML/CSS/JS** — Clean, responsive web interface

### 🔐 Auth0 — Secure Authentication
Secure user authentication with social sign-in, MFA, and passwordless login. Users can securely log in to save their preferences and track their personal food waste reduction journey.

### ☁️ Cloudflare Workers AI — Serverless AI Inference
Deployed our freshness analysis model on Cloudflare's edge network for ultra-low latency inference. Uses LLaVA vision model for enhanced food analysis and Llama 3 for recipe generation—all serverless and globally distributed.

### 🖥️ DigitalOcean Gradient™ AI — GPU-Accelerated ML
Leverages DigitalOcean's GPU Droplets and serverless inference for high-performance YOLO + ResNet model execution. Enables real-time processing with NVIDIA GPU acceleration.

### 🔊 ElevenLabs — Voice Alerts
Natural, human-sounding voice notifications alert users when food needs attention. Emotionally expressive alerts ("Your bananas are perfect for banana bread today!") make the experience delightful and actionable.

### 🤖 Generative AI — Smart Recipe Suggestions
Multi-provider AI recipe engine using:
- **OpenAI GPT-4** — Intelligent recipe generation
- **Anthropic Claude** — Contextual cooking suggestions
- **Google Gemini** — Alternative AI recommendations

Recipes are scored by a waste-reduction algorithm:
$$\text{Score}(r) = \alpha \cdot \text{Overlap}(r) + \beta \cdot \sum \text{Urgency}(i) - \gamma \cdot \text{Missing}(r)$$

### 🍃 MongoDB Atlas — Cloud Database
Stores detection history, user preferences, and analytics:
- Time-series freshness tracking
- Aggregated waste statistics  
- User preference persistence
- Environmental impact metrics

### ⛓️ Solana — Blockchain Waste Tracking
Decentralized, immutable record of food waste prevention:
- On-chain waste prevention records
- EcoToken rewards for sustainable behavior
- Transparent community leaderboard
- Verified environmental impact claims

---

## 🖥️ Try It

```bash
cd backend
pip install flask flask-cors torch torchvision ultralytics opencv-python pillow numpy
python web_app.py
```

Then open **http://localhost:5000** in your browser.

### Environment Variables (Optional)

```bash
# Auth0
export AUTH0_DOMAIN="your-tenant.auth0.com"
export AUTH0_CLIENT_ID="your-client-id"

# Cloudflare
# Deploy: cd integrations && npx wrangler deploy

# DigitalOcean
export DIGITALOCEAN_API_TOKEN="your-do-token"

# ElevenLabs
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# GenAI (any of these)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# MongoDB Atlas
export MONGODB_URI="mongodb+srv://..."

# Solana
export SOLANA_RPC_URL="https://api.devnet.solana.com"
```

---

## 📁 Project Structure

```
ecoshelf/
├── backend/
│   ├── web_app.py              # Main Flask application
│   ├── simple_app.py           # Desktop camera app
│   ├── model/
│   │   └── ripe_detector.pth   # Freshness classification model
│   └── integrations/
│       ├── auth0_config.py     # 🔐 Auth0 authentication
│       ├── cloudflare_worker.js # ☁️ Cloudflare Workers AI
│       ├── wrangler.toml       # Cloudflare deployment config
│       ├── digitalocean_gradient.py # 🖥️ DO Gradient AI
│       ├── elevenlabs_voice.py # 🔊 Voice alerts
│       ├── genai_recipes.py    # 🤖 AI recipe suggestions
│       ├── mongodb_atlas.py    # 🍃 MongoDB database
│       └── solana_tracker.py   # ⛓️ Blockchain tracking
└── frontend/                   # React frontend (optional)
```

---

## 🏆 MLH Prizes Targeted

| Prize | Technology | Integration |
|-------|------------|-------------|
| 🔐 Best Use of Auth0 | Auth0 | Secure user authentication |
| ☁️ Best AI Application Built with Cloudflare | Workers AI | Serverless freshness inference |
| 🖥️ Best Use of DigitalOcean Gradient™ AI | Gradient | GPU-accelerated ML |
| 🔊 Best Use of ElevenLabs | ElevenLabs | Voice freshness alerts |
| 🤖 Best Use of Gen AI | OpenAI/Claude/Gemini | Recipe suggestions |
| 🍃 Best Use of MongoDB Atlas | MongoDB | Detection history & analytics |
| ⛓️ Best Use of Solana | Solana | Blockchain waste tracking |

---

**Built with 💚 for reducing food waste, one fridge at a time.**
