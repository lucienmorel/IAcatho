"""
Script pour créer l'index EN LOCAL puis le déployer
Résout le problème de RAM sur Render (512MB)
"""

import pickle
import numpy as np
from pathlib import Path

# pip install pypdf sentence-transformers

class IndexeurPDF:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("🔄 Chargement du modèle (peut prendre 1 minute)...")
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        print("✅ Modèle chargé !")
    
    def extraire_texte_pdf(self, chemin_pdf):
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
    
    def creer_index(self, chemin_pdf, fichier_index="index_pdf.pkl"):
        """Crée l'index complet"""
        print("\n" + "="*60)
        print("🚀 CRÉATION DE L'INDEX")
        print("="*60 + "\n")
        
        # Extraction et découpage
        pages_texte = self.extraire_texte_pdf(chemin_pdf)
        chunks, metadata = self.decouper_en_chunks(pages_texte)
        
        # Calcul des embeddings
        print("\n🧮 Calcul des embeddings (1-3 minutes selon la taille)...")
        embeddings = self.model.encode(chunks, show_progress_bar=True, batch_size=32)
        
        # Sauvegarde
        print(f"\n💾 Sauvegarde dans {fichier_index}...")
        with open(fichier_index, 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': metadata,
                'model_name': self.model._model_card_vars.get('model_name', 'unknown')
            }, f)
        
        taille_mo = Path(fichier_index).stat().st_size / (1024 * 1024)
        
        print("\n" + "="*60)
        print("✅ INDEX CRÉÉ AVEC SUCCÈS !")
        print("="*60)
        print(f"📊 Statistiques :")
        print(f"   - Chunks créés : {len(chunks)}")
        print(f"   - Taille de l'index : {taille_mo:.1f} MB")
        print(f"   - Fichier : {fichier_index}")
        print("\n📤 Prochaine étape : Ajouter cet index à votre repo GitHub")
        print("   git add index_pdf.pkl")
        print("   git commit -m 'Ajout index pré-calculé'")
        print("   git push")
        print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("❌ Usage: python create_index.py mon_fichier.pdf")
        print("\nExemple: python create_index.py mon_cours.pdf")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    if not Path(pdf_file).exists():
        print(f"❌ Fichier non trouvé : {pdf_file}")
        sys.exit(1)
    
    indexeur = IndexeurPDF()
    indexeur.creer_index(pdf_file, "index_pdf.pkl")