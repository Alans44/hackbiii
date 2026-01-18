"""
EcoShelf - Smart Fridge Food Detector
A clean web app that displays camera feed inside a fridge graphic with detection results.
"""

from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import threading
import time
import os
import requests
from collections import deque

app = Flask(__name__)
CORS(app)

# ============== CONFIGURATION ==============
PRODUCE_CLASSES = ['apple', 'banana', 'orange', 'broccoli', 'carrot']
BOTTLE_CLASSES = ['bottle']
SNACK_CLASSES = ['cake', 'donut']
SANDWICH_CLASSES = ['sandwich', 'hot dog']
PIZZA_CLASSES = ['pizza']
ALL_CLASSES = PRODUCE_CLASSES + BOTTLE_CLASSES + SNACK_CLASSES + SANDWICH_CLASSES + PIZZA_CLASSES

FRESH_THRESHOLD = 70
MODERATE_THRESHOLD = 40

# Colors (BGR for OpenCV)
COLORS = {
    'fresh': (0, 255, 0),
    'moderate': (0, 165, 255),
    'spoiling': (0, 0, 255),
    'bottle': (255, 191, 0),
    'snack': (203, 192, 255),
    'sandwich': (0, 255, 255),
    'pizza': (0, 128, 255),
}

# Global state
detections_list = deque(maxlen=20)  # Keep last 20 detections
detection_lock = threading.Lock()

# ============== HTML TEMPLATE ==============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcoShelf - Smart Food Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: white;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        header {
            text-align: center;
            padding: 20px 0 30px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .logo-icon {
            font-size: 3rem;
        }
        
        h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4ade80, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tagline {
            color: #94a3b8;
            font-size: 1.1rem;
            font-weight: 300;
        }
        
        /* Main Layout */
        .main-content {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 30px;
            align-items: start;
        }
        
        /* Fridge Container */
        .fridge-container {
            position: relative;
            display: flex;
            justify-content: center;
        }
        
        .fridge {
            position: relative;
            background: linear-gradient(180deg, #e8e8e8 0%, #d0d0d0 100%);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.4),
                inset 0 2px 0 rgba(255, 255, 255, 0.3),
                inset 0 -2px 0 rgba(0, 0, 0, 0.1);
        }
        
        .fridge-handle {
            position: absolute;
            right: -15px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 120px;
            background: linear-gradient(90deg, #888, #aaa, #888);
            border-radius: 6px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .fridge-inner {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-radius: 12px;
            padding: 15px;
            box-shadow: inset 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        
        .camera-feed {
            width: 100%;
            max-width: 720px;
            border-radius: 8px;
            display: block;
        }
        
        .fridge-light {
            position: absolute;
            top: 30px;
            left: 50%;
            transform: translateX(-50%);
            width: 60%;
            height: 4px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 200, 0.6), transparent);
            border-radius: 2px;
            filter: blur(2px);
        }
        
        /* Detection Panel */
        .detection-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        
        .panel-header {
            background: linear-gradient(90deg, #4ade80, #22d3ee);
            padding: 20px;
            text-align: center;
        }
        
        .panel-header h2 {
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        .detection-list {
            padding: 15px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .detection-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border-left: 4px solid;
            transition: transform 0.2s, background 0.2s;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .detection-item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        .detection-item.fresh { border-color: #4ade80; }
        .detection-item.moderate { border-color: #fb923c; }
        .detection-item.spoiling { border-color: #ef4444; }
        .detection-item.bottle { border-color: #38bdf8; }
        .detection-item.snack { border-color: #f472b6; }
        .detection-item.sandwich { border-color: #facc15; }
        .detection-item.pizza { border-color: #fb923c; }
        
        .item-icon {
            font-size: 2rem;
            width: 50px;
            text-align: center;
        }
        
        .item-details {
            flex: 1;
        }
        
        .item-name {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 4px;
        }
        
        .item-status {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        
        .item-freshness {
            text-align: right;
        }
        
        .freshness-value {
            font-size: 1.5rem;
            font-weight: 700;
        }
        
        .freshness-value.fresh { color: #4ade80; }
        .freshness-value.moderate { color: #fb923c; }
        .freshness-value.spoiling { color: #ef4444; }
        
        .freshness-label {
            font-size: 0.75rem;
            color: #64748b;
            text-transform: uppercase;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #64748b;
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            opacity: 0.5;
        }
        
        /* Legend */
        .legend {
            display: flex;
            justify-content: center;
            gap: 25px;
            padding: 20px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            color: #94a3b8;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .legend-dot.fresh { background: #4ade80; }
        .legend-dot.moderate { background: #fb923c; }
        .legend-dot.spoiling { background: #ef4444; }
        
        /* Recipe Section */
        .recipe-section {
            margin-top: 40px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        
        .recipe-header {
            background: linear-gradient(90deg, #f472b6, #fb923c);
            padding: 20px;
            text-align: center;
        }
        
        .recipe-header h2 {
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        .recipe-content {
            padding: 25px;
        }
        
        .ingredient-input-section {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .ingredient-input {
            flex: 1;
            min-width: 200px;
            padding: 12px 18px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            color: white;
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .ingredient-input:focus {
            border-color: #4ade80;
        }
        
        .ingredient-input::placeholder {
            color: #64748b;
        }
        
        .add-btn, .generate-btn {
            padding: 12px 25px;
            border: none;
            border-radius: 12px;
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .add-btn {
            background: linear-gradient(135deg, #4ade80, #22d3ee);
            color: #0f172a;
        }
        
        .generate-btn {
            background: linear-gradient(135deg, #f472b6, #fb923c);
            color: #0f172a;
        }
        
        .add-btn:hover, .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        }
        
        .add-btn:disabled, .generate-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .ingredient-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            min-height: 40px;
        }
        
        .ingredient-tag {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: rgba(74, 222, 128, 0.2);
            border: 1px solid #4ade80;
            border-radius: 20px;
            font-size: 0.9rem;
            animation: tagPop 0.2s ease-out;
        }
        
        @keyframes tagPop {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        .ingredient-tag .remove-tag {
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        
        .ingredient-tag .remove-tag:hover {
            opacity: 1;
        }
        
        .recipe-results {
            margin-top: 20px;
        }
        
        .recipe-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #f472b6;
            animation: slideIn 0.3s ease-out;
        }
        
        .recipe-card h3 {
            color: #f472b6;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }
        
        .recipe-card .recipe-time {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 15px;
        }
        
        .recipe-card .recipe-ingredients {
            margin-bottom: 15px;
        }
        
        .recipe-card .recipe-ingredients strong {
            color: #4ade80;
        }
        
        .recipe-card .recipe-steps {
            color: #cbd5e1;
            line-height: 1.6;
        }
        
        .recipe-card .recipe-steps ol {
            padding-left: 20px;
        }
        
        .recipe-card .recipe-steps li {
            margin-bottom: 8px;
        }
        
        .loading-spinner {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 30px;
            color: #94a3b8;
        }
        
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: #f472b6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .use-detected-btn {
            padding: 8px 15px;
            background: rgba(74, 222, 128, 0.2);
            border: 1px solid #4ade80;
            border-radius: 8px;
            color: #4ade80;
            font-size: 0.85rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .use-detected-btn:hover {
            background: rgba(74, 222, 128, 0.3);
        }
        
        .quick-actions {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 30px;
            color: #64748b;
            font-size: 0.9rem;
        }
        
        /* Scrollbar */
        .detection-list::-webkit-scrollbar {
            width: 6px;
        }
        
        .detection-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        
        .detection-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        /* Responsive */
        @media (max-width: 1100px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .detection-panel {
                max-width: 500px;
                margin: 0 auto;
            }
        }
        
        @media (max-width: 600px) {
            h1 {
                font-size: 2rem;
            }
            
            .fridge {
                padding: 10px;
            }
            
            .camera-feed {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <span class="logo-icon">🥬</span>
                <h1>EcoShelf</h1>
                <span class="logo-icon">🍎</span>
            </div>
            <p class="tagline">Smart Food Monitoring • Reduce Waste • Stay Fresh</p>
        </header>
        
        <div class="main-content">
            <div class="fridge-container">
                <div class="fridge">
                    <div class="fridge-handle"></div>
                    <div class="fridge-inner">
                        <div class="fridge-light"></div>
                        <img src="/video_feed" alt="Camera Feed" class="camera-feed">
                    </div>
                </div>
            </div>
            
            <div class="detection-panel">
                <div class="panel-header">
                    <h2>📋 Detected Items</h2>
                </div>
                <div class="detection-list" id="detectionList">
                    <div class="empty-state">
                        <div class="empty-state-icon">📷</div>
                        <p>Point camera at food items<br>to see freshness analysis</p>
                    </div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot fresh"></div>
                        <span>Fresh (70%+)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot moderate"></div>
                        <span>Moderate (40-70%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot spoiling"></div>
                        <span>Spoiling (&lt;40%)</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recipe Section -->
        <div class="recipe-section">
            <div class="recipe-header">
                <h2>🍳 Recipe Generator</h2>
            </div>
            <div class="recipe-content">
                <div class="quick-actions">
                    <button class="use-detected-btn" onclick="useDetectedItems()">
                        ✨ Use Detected Items
                    </button>
                </div>
                
                <div class="ingredient-input-section">
                    <input type="text" class="ingredient-input" id="ingredientInput" 
                           placeholder="Type an ingredient (e.g., banana, eggs, milk...)" 
                           onkeypress="handleIngredientKeypress(event)">
                    <button class="add-btn" onclick="addIngredient()">+ Add</button>
                    <button class="generate-btn" onclick="generateRecipe()" id="generateBtn">
                        🍳 Generate Recipe
                    </button>
                </div>
                
                <div class="ingredient-tags" id="ingredientTags">
                    <!-- Tags will be added here -->
                </div>
                
                <div class="recipe-results" id="recipeResults">
                    <!-- Recipe will appear here -->
                </div>
            </div>
        </div>
        
        <footer>
            <p>🌱 EcoShelf — Helping reduce food waste, one shelf at a time</p>
        </footer>
    </div>
    
    <script>
        const ICONS = {
            'apple': '🍎',
            'banana': '🍌',
            'orange': '🍊',
            'broccoli': '🥦',
            'carrot': '🥕',
            'bottle': '🍼',
            'cake': '🍰',
            'donut': '🍩',
            'sandwich': '🥪',
            'hot dog': '🌭',
            'pizza': '🍕'
        };
        
        function getStatusClass(item) {
            if (item.freshness !== null) {
                if (item.freshness >= 70) return 'fresh';
                if (item.freshness >= 40) return 'moderate';
                return 'spoiling';
            }
            return item.category;
        }
        
        function getStatusText(item) {
            if (item.freshness !== null) {
                if (item.freshness >= 70) return `Fresh • Good for ~${Math.floor((item.freshness-50)/10 + 3)} days`;
                if (item.freshness >= 40) return `Moderate • Use within ~${Math.floor((item.freshness-30)/10 + 1)} days`;
                return 'Spoiling • Use immediately';
            }
            return item.status;
        }
        
        function updateDetections() {
            fetch('/detections')
                .then(response => response.json())
                .then(data => {
                    const list = document.getElementById('detectionList');
                    
                    if (data.length === 0) {
                        list.innerHTML = `
                            <div class="empty-state">
                                <div class="empty-state-icon">📷</div>
                                <p>Point camera at food items<br>to see freshness analysis</p>
                            </div>
                        `;
                        return;
                    }
                    
                    list.innerHTML = data.map(item => {
                        const statusClass = getStatusClass(item);
                        const icon = ICONS[item.name.toLowerCase()] || '🍽️';
                        const statusText = getStatusText(item);
                        
                        let freshnessHtml = '';
                        if (item.freshness !== null) {
                            freshnessHtml = `
                                <div class="item-freshness">
                                    <div class="freshness-value ${statusClass}">${Math.round(item.freshness)}%</div>
                                    <div class="freshness-label">Freshness</div>
                                </div>
                            `;
                        } else {
                            freshnessHtml = `
                                <div class="item-freshness">
                                    <div class="freshness-value" style="color: #94a3b8; font-size: 1rem;">✓</div>
                                    <div class="freshness-label">Detected</div>
                                </div>
                            `;
                        }
                        
                        return `
                            <div class="detection-item ${statusClass}">
                                <div class="item-icon">${icon}</div>
                                <div class="item-details">
                                    <div class="item-name">${item.name}</div>
                                    <div class="item-status">${statusText}</div>
                                </div>
                                ${freshnessHtml}
                            </div>
                        `;
                    }).join('');
                })
                .catch(err => console.error('Error fetching detections:', err));
        }
        
        // Update detections every 500ms
        setInterval(updateDetections, 500);
        updateDetections();
        
        // ============== RECIPE FUNCTIONALITY ==============
        let ingredients = [];
        
        function handleIngredientKeypress(event) {
            if (event.key === 'Enter') {
                addIngredient();
            }
        }
        
        function addIngredient() {
            const input = document.getElementById('ingredientInput');
            const value = input.value.trim().toLowerCase();
            
            if (value && !ingredients.includes(value)) {
                ingredients.push(value);
                renderIngredientTags();
            }
            input.value = '';
            input.focus();
        }
        
        function removeIngredient(ingredient) {
            ingredients = ingredients.filter(i => i !== ingredient);
            renderIngredientTags();
        }
        
        function renderIngredientTags() {
            const container = document.getElementById('ingredientTags');
            
            if (ingredients.length === 0) {
                container.innerHTML = '<span style="color: #64748b; font-size: 0.9rem;">No ingredients added yet. Add some above or use detected items!</span>';
                return;
            }
            
            container.innerHTML = ingredients.map(ing => `
                <div class="ingredient-tag">
                    <span>${ing}</span>
                    <span class="remove-tag" onclick="removeIngredient('${ing}')">✕</span>
                </div>
            `).join('');
        }
        
        function useDetectedItems() {
            fetch('/detections')
                .then(response => response.json())
                .then(data => {
                    data.forEach(item => {
                        const name = item.name.toLowerCase();
                        if (!ingredients.includes(name)) {
                            ingredients.push(name);
                        }
                    });
                    renderIngredientTags();
                });
        }
        
        function generateRecipe() {
            if (ingredients.length === 0) {
                alert('Please add some ingredients first!');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            const results = document.getElementById('recipeResults');
            
            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            
            results.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <span>Creating delicious recipes from your ingredients...</span>
                </div>
            `;
            
            // Find urgent items (low freshness from detections)
            fetch('/detections')
                .then(response => response.json())
                .then(detections => {
                    const urgentItems = detections
                        .filter(d => d.freshness !== null && d.freshness < 50)
                        .map(d => d.name.toLowerCase());
                    
                    return fetch('/generate-recipe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            ingredients: ingredients,
                            urgentItems: urgentItems
                        })
                    });
                })
                .then(response => response.json())
                .then(data => {
                    btn.disabled = false;
                    btn.textContent = '🍳 Generate Recipe';
                    
                    if (data.error) {
                        results.innerHTML = `<p style="color: #ef4444;">Error: ${data.error}</p>`;
                        return;
                    }
                    
                    results.innerHTML = data.recipes.map(recipe => `
                        <div class="recipe-card">
                            <h3>${recipe.name}</h3>
                            <div class="recipe-time">⏱️ ${recipe.time}</div>
                            <div class="recipe-ingredients">
                                <strong>Ingredients:</strong> ${recipe.ingredients.join(', ')}
                            </div>
                            <div class="recipe-steps">
                                <strong>Instructions:</strong>
                                <ol>
                                    ${recipe.steps.map(step => `<li>${step}</li>`).join('')}
                                </ol>
                            </div>
                        </div>
                    `).join('');
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.textContent = '🍳 Generate Recipe';
                    results.innerHTML = `<p style="color: #ef4444;">Failed to generate recipe. Please try again.</p>`;
                    console.error('Recipe error:', err);
                });
        }
        
        // Initialize ingredient tags
        renderIngredientTags();
    </script>
</body>
</html>
'''

# ============== FRESHNESS MODEL ==============
class FreshDetector(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FreshDetector, self).__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Global models
yolo_model = None
fresh_model = None
device = None
transform = None
camera = None


def load_models():
    global yolo_model, fresh_model, device, transform
    
    print("Loading YOLO model...")
    yolo_model = YOLO("yolov8n.pt")
    print("✓ YOLO model loaded")
    
    print("Loading freshness model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        fresh_model = FreshDetector()
        fresh_model.load_state_dict(torch.load("./model/ripe_detector.pth", map_location=device))
        fresh_model.to(device)
        fresh_model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✓ Freshness model loaded")
    except Exception as e:
        print(f"⚠ Freshness model not available: {e}")
        fresh_model = None


def get_freshness_score(image_bgr):
    if fresh_model is None:
        return None
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = fresh_model(image_tensor)
        probability = torch.sigmoid(output).item()
    return probability * 100


def generate_frames():
    global camera, detections_list
    
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    skip_frames = 2
    last_results = []
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame
        frame = cv2.flip(frame, -1)
        
        frame_count += 1
        current_detections = []
        
        # Run detection
        if frame_count % skip_frames == 0:
            results = yolo_model.predict(frame, conf=0.5, verbose=False, imgsz=480)
            last_results = results
            
            for result in results:
                boxes = result.boxes
                class_names = yolo_model.names
                
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = class_names[cls_id]
                    confidence = box.conf.item()
                    
                    if class_name.lower() not in ALL_CLASSES:
                        continue
                    
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Determine category and get freshness if produce
                    freshness = None
                    category = 'other'
                    status = 'Detected'
                    color = (200, 200, 200)
                    
                    if class_name.lower() in PRODUCE_CLASSES:
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            freshness = get_freshness_score(cropped)
                            if freshness is not None:
                                if freshness >= FRESH_THRESHOLD:
                                    category = 'fresh'
                                    color = COLORS['fresh']
                                elif freshness >= MODERATE_THRESHOLD:
                                    category = 'moderate'
                                    color = COLORS['moderate']
                                else:
                                    category = 'spoiling'
                                    color = COLORS['spoiling']
                    elif class_name.lower() in BOTTLE_CLASSES:
                        category = 'bottle'
                        color = COLORS['bottle']
                        status = 'Water/Plastic Bottle'
                    elif class_name.lower() in SNACK_CLASSES:
                        category = 'snack'
                        color = COLORS['snack']
                        status = 'Baked Good'
                    elif class_name.lower() in SANDWICH_CLASSES:
                        category = 'sandwich'
                        color = COLORS['sandwich']
                        status = 'Sandwich'
                    elif class_name.lower() in PIZZA_CLASSES:
                        category = 'pizza'
                        color = COLORS['pizza']
                        status = 'Pizza'
                    
                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = class_name.capitalize()
                    if freshness is not None:
                        label += f" {freshness:.0f}%"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    current_detections.append({
                        'name': class_name.capitalize(),
                        'freshness': freshness,
                        'category': category,
                        'status': status,
                        'confidence': confidence
                    })
            
            # Update global detections
            with detection_lock:
                detections_list.clear()
                detections_list.extend(current_detections)
        else:
            # Draw previous detections
            for result in last_results:
                boxes = result.boxes
                class_names = yolo_model.names
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = class_names[cls_id]
                    if class_name.lower() not in ALL_CLASSES:
                        continue
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ============== ROUTES ==============
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    with detection_lock:
        return jsonify(list(detections_list))


@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    """Generate recipes based on ingredients using GenAI or fallback"""
    data = request.json
    ingredients = data.get('ingredients', [])
    urgent_items = data.get('urgentItems', [])
    
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    # Try OpenAI first if API key is available
    openai_key = os.environ.get('OPENAI_API_KEY', '')
    
    if openai_key:
        try:
            recipes = generate_with_openai(ingredients, urgent_items, openai_key)
            return jsonify({'recipes': recipes, 'source': 'openai'})
        except Exception as e:
            print(f"OpenAI error: {e}")
    
    # Fallback to built-in recipes
    recipes = generate_fallback_recipes(ingredients, urgent_items)
    return jsonify({'recipes': recipes, 'source': 'fallback'})


def generate_with_openai(ingredients, urgent_items, api_key):
    """Generate recipes using OpenAI API"""
    prompt = f"""I have these ingredients: {', '.join(ingredients)}.
{f"These need to be used soon: {', '.join(urgent_items)}." if urgent_items else ""}

Generate 2 quick recipes (under 30 minutes). For each, respond in this exact JSON format:
[
  {{
    "name": "Recipe Name",
    "time": "X minutes",
    "ingredients": ["ingredient 1", "ingredient 2"],
    "steps": ["Step 1", "Step 2", "Step 3"]
  }}
]
Only return the JSON array, nothing else."""

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful chef focused on reducing food waste. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        },
        timeout=30
    )
    
    if response.status_code == 200:
        import json
        content = response.json()['choices'][0]['message']['content']
        # Parse JSON from response
        return json.loads(content)
    else:
        raise Exception(f"OpenAI API error: {response.status_code}")


def generate_fallback_recipes(ingredients, urgent_items):
    """Generate recipes using built-in logic when no API available"""
    recipes = []
    ing_lower = [i.lower() for i in ingredients]
    
    # Recipe database with variations
    recipe_db = [
        {
            'name': '🍌 Banana Smoothie',
            'requires': ['banana'],
            'time': '5 minutes',
            'ingredients': ['banana', 'milk or water', 'honey (optional)', 'ice'],
            'steps': [
                'Peel and slice the banana',
                'Add banana and 1 cup liquid to blender',
                'Blend until smooth',
                'Add honey to taste and serve cold'
            ]
        },
        {
            'name': '🍎 Apple Cinnamon Bites',
            'requires': ['apple'],
            'time': '10 minutes',
            'ingredients': ['apple', 'cinnamon', 'honey or peanut butter'],
            'steps': [
                'Slice apple into thin rounds or wedges',
                'Sprinkle with cinnamon',
                'Drizzle with honey or serve with peanut butter',
                'Enjoy as a healthy snack!'
            ]
        },
        {
            'name': '🥦 Quick Veggie Stir Fry',
            'requires': ['broccoli', 'carrot'],
            'time': '15 minutes',
            'ingredients': ['broccoli', 'carrot', 'oil', 'soy sauce', 'garlic'],
            'steps': [
                'Chop broccoli into florets, slice carrots',
                'Heat oil in pan over high heat',
                'Add garlic, then vegetables',
                'Stir fry 5-7 minutes, add soy sauce to taste'
            ]
        },
        {
            'name': '🥗 Fresh Garden Salad',
            'requires': ['carrot'],
            'time': '8 minutes',
            'ingredients': ['carrot', 'lettuce (optional)', 'olive oil', 'lemon juice'],
            'steps': [
                'Grate or julienne the carrots',
                'Combine with any greens you have',
                'Make dressing with olive oil and lemon',
                'Toss and season with salt and pepper'
            ]
        },
        {
            'name': '🍊 Citrus Energy Boost',
            'requires': ['orange'],
            'time': '5 minutes',
            'ingredients': ['orange', 'honey', 'water'],
            'steps': [
                'Squeeze fresh orange juice',
                'Mix with a glass of cold water',
                'Add honey to taste',
                'Perfect refreshing drink!'
            ]
        },
        {
            'name': '🥪 Loaded Sandwich',
            'requires': ['sandwich', 'bread'],
            'time': '10 minutes',
            'ingredients': ['bread', 'any vegetables', 'cheese', 'condiments'],
            'steps': [
                'Toast bread if desired',
                'Layer with available vegetables',
                'Add cheese and favorite condiments',
                'Press together and slice diagonally'
            ]
        },
        {
            'name': '🍕 Pizza Refresh',
            'requires': ['pizza'],
            'time': '8 minutes',
            'ingredients': ['leftover pizza', 'olive oil', 'fresh herbs'],
            'steps': [
                'Heat skillet over medium heat',
                'Add a drizzle of olive oil',
                'Place pizza slice in pan, cover with lid',
                'Cook 3-4 min until crispy and cheese melts'
            ]
        },
        {
            'name': '🍩 Sweet Treat Parfait',
            'requires': ['donut', 'cake'],
            'time': '5 minutes',
            'ingredients': ['donut or cake', 'yogurt', 'fresh fruit'],
            'steps': [
                'Crumble donut or cake into pieces',
                'Layer with yogurt in a glass',
                'Top with any fresh fruit',
                'A delicious dessert parfait!'
            ]
        },
        {
            'name': '🥤 Fruit Infused Water',
            'requires': ['bottle'],
            'time': '2 minutes',
            'ingredients': ['water bottle', 'any fruit', 'mint (optional)'],
            'steps': [
                'Slice any available fruit',
                'Add to your water bottle',
                'Add mint leaves for extra freshness',
                'Refrigerate 30 min for best flavor'
            ]
        }
    ]
    
    # Find matching recipes
    for recipe in recipe_db:
        required = recipe['requires']
        if any(req in ing_lower for req in required):
            recipes.append({
                'name': recipe['name'],
                'time': recipe['time'],
                'ingredients': recipe['ingredients'],
                'steps': recipe['steps']
            })
    
    # If no matches, provide a generic recipe
    if not recipes:
        recipes.append({
            'name': '🍳 Simple Ingredient Mix',
            'time': '15 minutes',
            'ingredients': ingredients[:5] + ['oil', 'salt', 'pepper'],
            'steps': [
                f'Prepare your ingredients: {", ".join(ingredients[:3])}',
                'Heat a pan with a little oil',
                'Cook ingredients together until done',
                'Season to taste and enjoy!'
            ]
        })
    
    # Prioritize urgent items
    if urgent_items:
        urgent_lower = [u.lower() for u in urgent_items]
        recipes.sort(key=lambda r: sum(1 for ing in r['ingredients'] if any(u in ing.lower() for u in urgent_lower)), reverse=True)
    
    return recipes[:3]  # Return max 3 recipes


# ============== MAIN ==============
if __name__ == '__main__':
    print("=" * 50)
    print("  🥬 EcoShelf - Smart Food Monitor 🍎")
    print("=" * 50)
    
    load_models()
    
    print("\n🌐 Starting web server...")
    print("   Open http://localhost:5000 in your browser")
    print("   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
