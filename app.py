"""
Moteur de recherche sémantique pour PDF - VERSION WEB
Backend FastAPI avec interface HTML
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
import re

# Installation requise :
# pip install pypdf sentence-transformers fastapi uvicorn python-multipart

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

class Question(BaseModel):
    question: str
    top_k: int = 3

class MoteurRecherchePDF:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print("🔄 Chargement du modèle d'embeddings...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.metadata = []  # Pour stocker les numéros de page
        print("✅ Modèle chargé !")
    
    def extraire_texte_pdf(self, chemin_pdf):
        """Extrait le texte du PDF avec les numéros de page"""
        from pypdf import PdfReader
        
        print(f"📄 Lecture du PDF : {chemin_pdf}")
        reader = PdfReader(chemin_pdf)
        pages_texte = []
        
        for i, page in enumerate(reader.pages):
            texte = page.extract_text()
            pages_texte.append({
                'numero_page': i + 1,
                'texte': texte
            })
        
        print(f"✅ {len(reader.pages)} pages extraites")
        return pages_texte
    
    def decouper_en_chunks(self, pages_texte, taille_chunk=500, overlap=100):
        """Découpe le texte en morceaux en conservant les numéros de page"""
        print(f"✂️  Découpage en chunks...")
        
        chunks = []
        metadata = []
        
        for page_data in pages_texte:
            page_num = page_data['numero_page']
            texte = page_data['texte']
            mots = texte.split()
            
            if not mots:
                continue
            
            chars_par_mot = len(texte) / len(mots)
            mots_par_chunk = int(taille_chunk / chars_par_mot) if chars_par_mot > 0 else 100
            overlap_mots = int(overlap / chars_par_mot) if chars_par_mot > 0 else 20
            
            for i in range(0, len(mots), mots_par_chunk - overlap_mots):
                chunk = ' '.join(mots[i:i + mots_par_chunk])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
                    metadata.append({'page': page_num})
        
        print(f"✅ {len(chunks)} chunks créés")
        return chunks, metadata
    
    def indexer_pdf(self, chemin_pdf, fichier_index="index_pdf.pkl"):
        """Indexe le PDF avec métadonnées de page"""
        pages_texte = self.extraire_texte_pdf(chemin_pdf)
        self.chunks, self.metadata = self.decouper_en_chunks(pages_texte)
        
        print("🧮 Calcul des embeddings...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        
        print(f"💾 Sauvegarde de l'index...")
        with open(fichier_index, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
        
        print(f"✅ Indexation terminée !")
        return len(self.chunks)
    
    def charger_index(self, fichier_index="index_pdf.pkl"):
        """Charge un index précédemment créé"""
        print(f"📂 Chargement de l'index...")
        with open(fichier_index, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.metadata = data.get('metadata', [{}] * len(self.chunks))
        print(f"✅ Index chargé : {len(self.chunks)} chunks")
    
    def rechercher(self, question, top_k=3):
        """Recherche avec numéro de page"""
        if self.embeddings is None:
            raise ValueError("Aucun index chargé")
        
        question_emb = self.model.encode([question])[0]
        
        similarities = np.dot(self.embeddings, question_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(question_emb)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        resultats = []
        for i, idx in enumerate(top_indices):
            resultats.append({
                'rang': i + 1,
                'score': float(similarities[idx]),
                'texte': self.chunks[idx],
                'page': self.metadata[idx].get('page', 'N/A') if idx < len(self.metadata) else 'N/A'
            })
        
        return resultats

# Initialisation FastAPI
app = FastAPI(title="Moteur de Recherche PDF")
moteur = MoteurRecherchePDF()
INDEX_FILE = "index_pdf.pkl"

# Charger l'index au démarrage si disponible
if os.path.exists(INDEX_FILE):
    try:
        moteur.charger_index(INDEX_FILE)
    except Exception as e:
        print(f"⚠️  Impossible de charger l'index : {e}")

@app.get("/", response_class=HTMLResponse)
async def interface():
    """Interface web principale"""
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔍 Moteur de Recherche PDF</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                text-align: center;
            }
            h1 { color: #667eea; font-size: 2.5em; margin-bottom: 10px; }
            .status {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                margin-top: 10px;
            }
            .status.ready { background: #d4edda; color: #155724; }
            .status.empty { background: #fff3cd; color: #856404; }
            
            .upload-section, .search-section {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .upload-zone {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background: #f8f9ff;
            }
            .upload-zone:hover { background: #eef1ff; border-color: #764ba2; }
            .upload-zone.dragover { background: #e0e7ff; border-color: #4338ca; }
            
            input[type="file"] { display: none; }
            
            .search-box {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            input[type="text"] {
                flex: 1;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 1em;
                transition: border 0.3s;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button {
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button:hover { transform: translateY(-2px); }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .results {
                margin-top: 30px;
            }
            .result-card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                border-left: 4px solid #667eea;
            }
            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .result-rank {
                background: #667eea;
                color: white;
                padding: 5px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
            }
            .result-page {
                background: #f0f0f0;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                color: #666;
            }
            .result-score {
                color: #28a745;
                font-weight: 600;
            }
            .result-text {
                line-height: 1.6;
                color: #333;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #667eea;
                font-weight: 600;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Moteur de Recherche PDF</h1>
                <div id="status" class="status empty">Aucun PDF indexé</div>
            </div>
            
            <div class="upload-section">
                <h2 style="margin-bottom: 20px;">📤 Indexer un PDF</h2>
                <div class="upload-zone" id="uploadZone">
                    <p style="font-size: 3em; margin-bottom: 10px;">📄</p>
                    <p style="font-size: 1.2em; color: #667eea; margin-bottom: 10px;">
                        Glissez votre PDF ici ou cliquez pour sélectionner
                    </p>
                    <p style="color: #999; font-size: 0.9em;">
                        Le fichier sera indexé pour permettre la recherche sémantique
                    </p>
                    <input type="file" id="fileInput" accept=".pdf">
                </div>
                <div id="uploadProgress" style="margin-top: 20px; display: none;"></div>
            </div>
            
            <div class="search-section">
                <h2 style="margin-bottom: 20px;">💬 Posez votre question</h2>
                <div class="search-box">
                    <input type="text" id="questionInput" 
                           placeholder="Ex: Quelle est la définition de...?"
                           disabled>
                    <button id="searchBtn" onclick="rechercher()" disabled>Rechercher</button>
                </div>
                <div id="results" class="results"></div>
            </div>
        </div>
        
        <script>
            const uploadZone = document.getElementById('uploadZone');
            const fileInput = document.getElementById('fileInput');
            const questionInput = document.getElementById('questionInput');
            const searchBtn = document.getElementById('searchBtn');
            const status = document.getElementById('status');
            const results = document.getElementById('results');
            
            // Vérifier le statut au chargement
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.indexed) {
                        activerRecherche(data.chunks_count);
                    }
                });
            
            // Upload par clic
            uploadZone.addEventListener('click', () => fileInput.click());
            
            // Drag & Drop
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    uploaderPDF(e.dataTransfer.files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploaderPDF(e.target.files[0]);
                }
            });
            
            async function uploaderPDF(file) {
                const progress = document.getElementById('uploadProgress');
                progress.style.display = 'block';
                progress.innerHTML = '<div class="spinner"></div><p class="loading">Indexation en cours... (1-2 minutes)</p>';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        progress.innerHTML = `<p style="color: #28a745; font-weight: 600;">✅ ${data.message}</p>`;
                        activerRecherche(data.chunks_count);
                    } else {
                        progress.innerHTML = `<p style="color: #dc3545;">❌ Erreur: ${data.detail}</p>`;
                    }
                } catch (error) {
                    progress.innerHTML = `<p style="color: #dc3545;">❌ Erreur: ${error.message}</p>`;
                }
            }
            
            function activerRecherche(chunks) {
                status.textContent = `✅ PDF indexé (${chunks} segments)`;
                status.className = 'status ready';
                questionInput.disabled = false;
                searchBtn.disabled = false;
                questionInput.focus();
            }
            
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') rechercher();
            });
            
            async function rechercher() {
                const question = questionInput.value.trim();
                if (!question) return;
                
                results.innerHTML = '<div class="spinner"></div><p class="loading">Recherche en cours...</p>';
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, top_k: 3 })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        afficherResultats(data.resultats);
                    } else {
                        results.innerHTML = `<p style="color: #dc3545;">❌ ${data.detail}</p>`;
                    }
                } catch (error) {
                    results.innerHTML = `<p style="color: #dc3545;">❌ Erreur: ${error.message}</p>`;
                }
            }
            
            function afficherResultats(resultats) {
                if (resultats.length === 0) {
                    results.innerHTML = '<p style="color: #999;">Aucun résultat trouvé.</p>';
                    return;
                }
                
                results.innerHTML = resultats.map(r => `
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-rank">#${r.rang}</span>
                            <span class="result-page">📄 Page ${r.page}</span>
                            <span class="result-score">Score: ${(r.score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="result-text">${r.texte.substring(0, 600)}${r.texte.length > 600 ? '...' : ''}</div>
                    </div>
                `).join('');
            }
        </script>
    </body>
    </html>
    """
    return html

@app.get("/status")
async def get_status():
    """Vérifie si un index est chargé"""
    return {
        "indexed": moteur.embeddings is not None,
        "chunks_count": len(moteur.chunks) if moteur.chunks else 0
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload et indexation du PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF")
    
    # Sauvegarder temporairement
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Indexer
        chunks_count = moteur.indexer_pdf(temp_path, INDEX_FILE)
        
        # Nettoyer
        os.remove(temp_path)
        
        return {
            "message": f"PDF indexé avec succès !",
            "chunks_count": chunks_count
        }
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(question: Question):
    """Effectue une recherche"""
    if moteur.embeddings is None:
        raise HTTPException(status_code=400, detail="Aucun PDF indexé")
    
    try:
        resultats = moteur.rechercher(question.question, question.top_k)
        return {"resultats": resultats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Démarrage du serveur...")
    print("📍 Ouvrez votre navigateur sur : http://localhost:8000")
    print("👉 Glissez-déposez votre PDF puis posez vos questions !\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)