import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
import tempfile
import io
from PIL import Image
import time
from datetime import datetime
import cv2
from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report
import base64

# Imports de vos modules
import sys
sys.path.append('src')
from autoencoder import create_autoencoder
from convlstm import create_tumor_predictor
from evaluate import compute_metrics, ModelEvaluator
from train import get_default_config

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🧠 Tumor Evolution Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Interface de Prédiction d'Évolution Tumorale\nPowered by Deep Learning & Streamlit"
    }
)

# CSS personnalisé avec thème moderne
st.markdown("""
<style>
    /* Variables CSS personnalisées */
    :root {
        --primary-color: #1e3d59;
        --secondary-color: #f5f7fa;
        --accent-color: #ff6b6b;
        --success-color: #51cf66;
        --warning-color: #ffd43b;
    }
    
    /* Style principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: #f0f8ff;
    }
    
    .success-message {
        background: linear-gradient(90deg, #51cf66, #40c057);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(90deg, #ffd43b, #fab005);
        color: #495057;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Animation pour les éléments de chargement */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-radius: 50%;
        border-top: 4px solid var(--primary-color);
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Styles pour les tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 10px 10px 0px 0px;
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 1.2rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Style pour les alertes personnalisées */
    .custom-alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #0d47a1;
    }
    
    .alert-success {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        color: #1b5e20;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        color: #e65100;
    }
    
    .alert-error {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'current_patient_data' not in st.session_state:
    st.session_state.current_patient_data = None

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🧠 Interface de Prédiction d'Évolution Tumorale</h1>
    <p>Analyse et prédiction basées sur l'Intelligence Artificielle</p>
    <p><em>AutoEncoder + ConvLSTM pour la prédiction temporelle</em></p>
</div>
""", unsafe_allow_html=True)

# Fonction utilitaires
@st.cache_data
def load_training_history():
    """Charge l'historique d'entraînement"""
    try:
        # AutoEncoder
        ae_history = pd.read_csv("outputs/logs/ae_loss_history.csv")
        # Predictor
        pred_history = pd.read_csv("outputs/logs/prediction_loss_history.csv")
        return ae_history, pred_history
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_evaluation_results():
    """Charge les résultats d'évaluation"""
    try:
        results = pd.read_csv("outputs/evaluation/evaluation_results.csv")
        return results
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    """Charge les modèles entraînés"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = get_default_config()
        
        # Charger l'autoencoder
        autoencoder = create_autoencoder(latent_dim=config['latent_dim'])
        ae_path = "outputs/models/autoencoder.pth"
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
            autoencoder.to(device)
            autoencoder.eval()
        else:
            st.error("❌ AutoEncoder non trouvé!")
            return None, None
        
        # Charger le predictor
        predictor = create_tumor_predictor(
            autoencoder,
            latent_dim=config['latent_dim'],
            use_mask=True
        )
        
        best_path = "outputs/models/best_predictor.pth"
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)
            predictor.load_state_dict(checkpoint['model_state_dict'])
            predictor.to(device)
            predictor.eval()
        else:
            st.error("❌ Modèle de prédiction non trouvé!")
            return None, None
            
        return autoencoder, predictor
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {e}")
        return None, None

def preprocess_nifti(nifti_data, target_size=(64, 128, 128)):
    """Préprocessing des données NIfTI"""
    # Redimensionnement
    zoom_factors = (
        target_size[0] / nifti_data.shape[0],
        target_size[1] / nifti_data.shape[1],
        target_size[2] / nifti_data.shape[2]
    )
    resized = ndimage.zoom(nifti_data, zoom_factors, order=1)
    
    # Normalisation Z-score
    if resized.std() > 0:
        resized = (resized - resized.mean()) / resized.std()
    
    return resized

def create_3d_visualization(volume, title="Volume 3D"):
    """Crée une visualisation 3D interactive"""
    # Sélectionner quelques slices représentatifs
    slices = []
    depth = volume.shape[0]
    slice_indices = [depth//6, depth//3, depth//2, 2*depth//3, 5*depth//6]
    
    for i in slice_indices:
        if i < depth:
            slices.append(volume[i])
    
    fig = make_subplots(
        rows=1, cols=len(slices),
        subplot_titles=[f"Slice {slice_indices[i]}" for i in range(len(slices))],
        specs=[[{"type": "heatmap"}]*len(slices)]
    )
    
    for i, slice_data in enumerate(slices):
        fig.add_trace(
            go.Heatmap(z=slice_data, colorscale='gray', showscale=(i==len(slices)-1)),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    return fig

def predict_tumor_evolution(autoencoder, predictor, t1_brain, t2_brain, t1_mask, t2_mask):
    """Effectue la prédiction d'évolution tumorale"""
    device = next(predictor.parameters()).device
    
    # Préprocessing
    t1_brain_proc = preprocess_nifti(t1_brain)
    t2_brain_proc = preprocess_nifti(t2_brain)
    t1_mask_proc = preprocess_nifti(t1_mask)
    t2_mask_proc = preprocess_nifti(t2_mask)
    
    # Conversion en tensors
    input_images = torch.stack([
        torch.from_numpy(t1_brain_proc[np.newaxis, ...]).float(),
        torch.from_numpy(t2_brain_proc[np.newaxis, ...]).float()
    ]).unsqueeze(0).to(device)  # (1, T, 1, D, H, W)
    
    input_masks = torch.stack([
        torch.from_numpy(t1_mask_proc[np.newaxis, ...]).float(),
        torch.from_numpy(t2_mask_proc[np.newaxis, ...]).float()
    ]).unsqueeze(0).to(device)   # (1, T, 1, D, H, W)
    
    # Prédiction
    with torch.no_grad():
        pred_logits = predictor(input_images, input_masks)
        pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)
    
    return pred_prob, pred_binary

def create_comparison_plot(t1_mask, t2_mask, pred_prob, pred_binary, slice_idx=None):
    """Crée un graphique de comparaison des timepoints"""
    if slice_idx is None:
        slice_idx = t1_mask.shape[0] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # T1 Mask
    axes[0, 0].imshow(t1_mask[slice_idx], cmap='Reds', alpha=0.8)
    axes[0, 0].set_title('Timepoint 1 - Masque Tumoral')
    axes[0, 0].axis('off')
    
    # T2 Mask
    axes[0, 1].imshow(t2_mask[slice_idx], cmap='Blues', alpha=0.8)
    axes[0, 1].set_title('Timepoint 2 - Masque Tumoral')
    axes[0, 1].axis('off')
    
    # Predicted T3 (probability)
    axes[1, 0].imshow(pred_prob[slice_idx], cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Timepoint 3 - Prédiction (Probabilité)')
    axes[1, 0].axis('off')
    
    # Predicted T3 (binary)
    axes[1, 1].imshow(pred_binary[slice_idx], cmap='Greens', alpha=0.8)
    axes[1, 1].set_title('Timepoint 3 - Prédiction (Binaire)')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Évolution Tumorale - Slice {slice_idx}', fontsize=16)
    plt.tight_layout()
    
    # Convertir en image pour Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def download_button_with_data(data, filename, label, mime_type="application/octet-stream"):
    """Crée un bouton de téléchargement avec données"""
    b64_data = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Sidebar pour la navigation
with st.sidebar:
    st.markdown("## 🎛️ Navigation")
    selected_tab = st.selectbox(
        "Choisissez une section:",
        ["📊 Dashboard", "📈 Statistiques", "🔮 Prédiction", "📋 Évaluation", "⚙️ Configuration", "📚 Documentation"]
    )
    
    st.markdown("---")
    st.markdown("## 📁 Informations Système")
    
    # Vérification des modèles
    if os.path.exists("outputs/models/best_predictor.pth"):
        st.success("✅ Modèles disponibles")
    else:
        st.error("❌ Modèles non trouvés")
    
    # Informations sur le device
    device_info = "🖥️ CPU"
    if torch.cuda.is_available():
        device_info = f"🚀 GPU: {torch.cuda.get_device_name()}"
    st.info(device_info)
    
    # Informations mémoire GPU
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        st.metric("Mémoire GPU", f"{memory_allocated:.2f} GB", f"Cache: {memory_cached:.2f} GB")
    
    st.markdown("---")
    st.markdown("### 📖 Guide d'utilisation")
    with st.expander("Aide", expanded=False):
        st.markdown("""
        **📊 Dashboard**: Vue d'ensemble des performances
        **📈 Statistiques**: Courbes d'entraînement et métriques
        **🔮 Prédiction**: Interface de prédiction interactive
        **📋 Évaluation**: Résultats détaillés sur le dataset test
        **⚙️ Configuration**: Paramètres et réglages
        **📚 Documentation**: Guide complet d'utilisation
        """)

# Contenu principal basé sur la sélection
if selected_tab == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Global")
    
    # Métriques principales en cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Charger les résultats d'évaluation
    eval_results = load_evaluation_results()
    
    if eval_results is not None:
        with col1:
            avg_dice = eval_results['dice'].mean()
            dice_std = eval_results['dice'].std()
            st.metric(
                label="🎯 Dice Score Moyen",
                value=f"{avg_dice:.3f}",
                delta=f"±{dice_std:.3f}",
                help="Coefficient de Dice moyen sur le dataset de test"
            )
        
        with col2:
            avg_iou = eval_results['iou'].mean()
            iou_std = eval_results['iou'].std()
            st.metric(
                label="🔲 IoU Moyen",
                value=f"{avg_iou:.3f}",
                delta=f"±{iou_std:.3f}",
                help="Intersection over Union moyen"
            )
        
        with col3:
            avg_sensitivity = eval_results['sensitivity'].mean()
            sens_std = eval_results['sensitivity'].std()
            st.metric(
                label="🔍 Sensibilité Moyenne",
                value=f"{avg_sensitivity:.3f}",
                delta=f"±{sens_std:.3f}",
                help="Sensibilité (Recall) moyenne"
            )
        
        with col4:
            n_patients = len(eval_results)
            best_dice = eval_results['dice'].max()
            st.metric(
                label="👥 Patients Évalués",
                value=str(n_patients),
                delta=f"Best: {best_dice:.3f}",
                help="Nombre total de patients dans l'évaluation"
            )
        
        st.markdown("---")
        
        # Graphiques de performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Distribution des Scores")
            
            # Créer des tabs pour différentes métriques
            metric_tabs = st.tabs(["Dice", "IoU", "Sensibilité", "Spécificité"])
            
            with metric_tabs[0]:
                fig_dice = px.histogram(
                    eval_results, 
                    x='dice', 
                    nbins=15,
                    title="Distribution du Dice Score",
                    color_discrete_sequence=['#667eea']
                )
                fig_dice.update_layout(showlegend=False)
                st.plotly_chart(fig_dice, use_container_width=True)
            
            with metric_tabs[1]:
                fig_iou = px.histogram(
                    eval_results, 
                    x='iou', 
                    nbins=15,
                    title="Distribution de l'IoU",
                    color_discrete_sequence=['#51cf66']
                )
                fig_iou.update_layout(showlegend=False)
                st.plotly_chart(fig_iou, use_container_width=True)
            
            with metric_tabs[2]:
                fig_sens = px.histogram(
                    eval_results, 
                    x='sensitivity', 
                    nbins=15,
                    title="Distribution de la Sensibilité",
                    color_discrete_sequence=['#ff6b6b']
                )
                fig_sens.update_layout(showlegend=False)
                st.plotly_chart(fig_sens, use_container_width=True)
            
            with metric_tabs[3]:
                fig_spec = px.histogram(
                    eval_results, 
                    x='specificity', 
                    nbins=15,
                    title="Distribution de la Spécificité",
                    color_discrete_sequence=['#ffd43b']
                )
                fig_spec.update_layout(showlegend=False)
                st.plotly_chart(fig_spec, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Performance par Patient")
            
            # Graphique interactif avec sélection de métrique
            metric_choice = st.selectbox(
                "Choisir la métrique à afficher:",
                ['dice', 'iou', 'sensitivity', 'specificity', 'precision']
            )
            
            fig_performance = px.bar(
                eval_results,
                x='patient_id',
                y=metric_choice,
                title=f"{metric_choice.capitalize()} par Patient",
                color=metric_choice,
                color_continuous_scale='viridis'
            )
            fig_performance.update_layout(
                xaxis_title="Patient",
                yaxis_title=metric_choice.capitalize(),
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_performance, use_container_width=True)
    
        # Analyse avancée
        st.markdown("### 📊 Analyse Avancée des Performances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matrice de corrélation
            correlation_metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'precision']
            if all(metric in eval_results.columns for metric in correlation_metrics):
                corr_matrix = eval_results[correlation_metrics].corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="Matrice de Corrélation des Métriques",
                    text_auto=True
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Box plot des métriques
            metrics_melted = pd.melt(
                eval_results[correlation_metrics], 
                var_name='Métrique', 
                value_name='Score'
            )
            
            fig_box = px.box(
                metrics_melted,
                x='Métrique',
                y='Score',
                title="Distribution des Métriques",
                color='Métrique'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
    else:
        st.markdown("""
        <div class="custom-alert alert-warning">
            ⚠️ <strong>Aucun résultat d'évaluation trouvé</strong><br>
            Veuillez d'abord lancer l'évaluation du modèle depuis l'onglet Évaluation.
        </div>
        """, unsafe_allow_html=True)
    
    # Section récente activité
    st.markdown("### 📋 Activité Récente")
    recent_activity = []
    
    # Vérifier les fichiers récents
    model_files = ["outputs/models/best_predictor.pth", "outputs/models/autoencoder.pth"]
    for file_path in model_files:
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            recent_activity.append({
                "Action": f"Sauvegarde {Path(file_path).name}",
                "Date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "Status": "✅ Terminé",
                "Taille": f"{os.path.getsize(file_path) / 1e6:.1f} MB"
            })
    
    if recent_activity:
        df_activity = pd.DataFrame(recent_activity)
        st.dataframe(df_activity, use_container_width=True)
    else:
        st.info("Aucune activité récente détectée.")

elif selected_tab == "📈 Statistiques":
    st.markdown("## 📈 Statistiques d'Entraînement")
    
    # Charger l'historique d'entraînement
    ae_history, pred_history = load_training_history()
    
    # Tabs pour différentes vues
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 AutoEncoder", "🧠 Prédicteur", "🆚 Comparaison", "📈 Métriques Temps Réel"])
    
    with tab1:
        if ae_history is not None:
            st.markdown("### 🔧 Entraînement AutoEncoder")
            
            # Contrôles interactifs
            col1, col2 = st.columns([3, 1])
            
            with col2:
                smooth_factor = st.slider("Lissage des courbes", 0.0, 1.0, 0.1, 0.05)
                show_points = st.checkbox("Afficher les points", True)
            
            with col1:
                fig_ae = make_subplots(specs=[[{"secondary_y": False}]])
                
                # Lissage optionnel
                if smooth_factor > 0:
                    from scipy.signal import savgol_filter
                    window_length = min(len(ae_history), max(3, int(len(ae_history) * smooth_factor)))
                    if window_length % 2 == 0:
                        window_length += 1
                    
                    train_smooth = savgol_filter(ae_history['train_loss'], window_length, 2)
                    val_smooth = savgol_filter(ae_history['val_loss'], window_length, 2)
                else:
                    train_smooth = ae_history['train_loss']
                    val_smooth = ae_history['val_loss']
                
                mode = 'lines+markers' if show_points else 'lines'
                
                fig_ae.add_trace(
                    go.Scatter(
                        x=ae_history['epoch'],
                        y=train_smooth,
                        mode=mode,
                        name='Train Loss',
                        line=dict(color='#ff6b6b', width=3)
                    )
                )
                fig_ae.add_trace(
                    go.Scatter(
                        x=ae_history['epoch'],
                        y=val_smooth,
                        mode=mode,
                        name='Validation Loss',
                        line=dict(color='#51cf66', width=3)
                    )
                )
                
                fig_ae.update_layout(
                    title="Courbes de Loss - AutoEncoder",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_ae, use_container_width=True)
            
            # Statistiques détaillées
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Loss Train Final", f"{ae_history['train_loss'].iloc[-1]:.6f}")
            with col2:
                st.metric("Loss Val Final", f"{ae_history['val_loss'].iloc[-1]:.6f}")
            with col3:
                best_epoch = ae_history.loc[ae_history['val_loss'].idxmin(), 'epoch']
                st.metric("Meilleure Epoch", f"{best_epoch}")
            with col4:
                improvement = (ae_history['train_loss'].iloc[0] - ae_history['train_loss'].iloc[-1]) / ae_history['train_loss'].iloc[0] * 100
                st.metric("Amélioration", f"{improvement:.1f}%")
        else:
            st.warning("⚠️ Historique AutoEncoder non trouvé")
    
    with tab2:
        if pred_history is not None:
            st.markdown("### 🧠 Entraînement Prédicteur")
            
            # Interface similaire pour le prédicteur
            col1, col2 = st.columns([3, 1])
            
            with col2:
                show_epochs = st.multiselect(
                    "Epochs à afficher:",
                    options=list(range(1, len(pred_history) + 1)),
                    default=list(range(1, min(len(pred_history) + 1, 11)))
                )
                log_scale = st.checkbox("Échelle logarithmique", False)
            
            with col1:
                if show_epochs:
                    filtered_history = pred_history[pred_history['epoch'].isin(show_epochs)]
                    
                    fig_pred = make_subplots(specs=[[{"secondary_y": False}]])
                    
                    fig_pred.add_trace(
                        go.Scatter(
                            x=filtered_history['epoch'],
                            y=filtered_history['train_loss'],
                            mode='lines+markers',
                            name='Train Loss',
                            line=dict(color='#ff6b6b', width=3)
                        )
                    )
                    fig_pred.add_trace(
                        go.Scatter(
                            x=filtered_history['epoch'],
                            y=filtered_history['val_loss'],
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='#51cf66', width=3)
                        )
                    )
                    
                    fig_pred.update_layout(
                        title="Courbes de Loss - Prédicteur",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        hovermode='x unified'
                    )
                    
                    if log_scale:
                        fig_pred.update_yaxes(type="log")
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            # Statistiques détaillées
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Loss Train Final", f"{pred_history['train_loss'].iloc[-1]:.6f}")
            with col2:
                st.metric("Loss Val Final", f"{pred_history['val_loss'].iloc[-1]:.6f}")
            with col3:
                best_epoch = pred_history.loc[pred_history['val_loss'].idxmin(), 'epoch']
                st.metric("Meilleure Epoch", f"{best_epoch}")
            with col4:
                improvement = (pred_history['train_loss'].iloc[0] - pred_history['train_loss'].iloc[-1]) / pred_history['train_loss'].iloc[0] * 100
                st.metric("Amélioration", f"{improvement:.1f}%")
        else:
            st.warning("⚠️ Historique Prédicteur non trouvé")
    
    with tab3:
        if ae_history is not None and pred_history is not None:
            st.markdown("### 🆚 Comparaison des Entraînements")
            
            fig_comp = make_subplots(specs=[[{"secondary_y": False}]])
            
            fig_comp.add_trace(
                go.Scatter(
                    x=ae_history['epoch'],
                    y=ae_history['val_loss'],
                    mode='lines',
                    name='AE Val Loss',
                    line=dict(color='#667eea', width=3)
                )
            )
            fig_comp.add_trace(
                go.Scatter(
                    x=pred_history['epoch'],
                    y=pred_history['val_loss'],
                    mode='lines',
                    name='Pred Val Loss',
                    line=dict(color='#764ba2', width=3)
                )
            )
            
            fig_comp.update_layout(
                title="Comparaison des Losses de Validation",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                hovermode='x unified'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("⚠️ Historiques incomplets pour la comparaison")
    
    with tab4:
        st.markdown("### 📈 Métriques Temps Réel")
        st.info("Cette section peut être étendue pour un monitoring en temps réel pendant l'entraînement. Pour l'instant, voici les historiques bruts.")
        
        if ae_history is not None:
            st.subheader("AutoEncoder")
            st.dataframe(ae_history)
        
        if pred_history is not None:
            st.subheader("Prédicteur")
            st.dataframe(pred_history)

elif selected_tab == "🔮 Prédiction":
    st.markdown("## 🔮 Prédiction d'Évolution Tumorale")
    
    # Chargement des modèles
    autoencoder, predictor = load_models()
    if autoencoder is None or predictor is None:
        st.stop()
    
    # Upload des fichiers NIfTI
    st.markdown("### 📤 Upload des Données du Patient")
    
    col1, col2 = st.columns(2)
    
    with col1:
        t1_brain_file = st.file_uploader("Timepoint 1 - Brain T1C (NIfTI)", type=["nii", "gz"])
        t1_mask_file = st.file_uploader("Timepoint 1 - Tumor Mask (NIfTI)", type=["nii", "gz"])
    
    with col2:
        t2_brain_file = st.file_uploader("Timepoint 2 - Brain T1C (NIfTI)", type=["nii", "gz"])
        t2_mask_file = st.file_uploader("Timepoint 2 - Tumor Mask (NIfTI)", type=["nii", "gz"])
    
    if t1_brain_file and t1_mask_file and t2_brain_file and t2_mask_file:
        # Chargement des données
        t1_brain = nib.load(io.BytesIO(t1_brain_file.read())).get_fdata()
        t1_mask = nib.load(io.BytesIO(t1_mask_file.read())).get_fdata()
        t2_brain = nib.load(io.BytesIO(t2_brain_file.read())).get_fdata()
        t2_mask = nib.load(io.BytesIO(t2_mask_file.read())).get_fdata()
        
        # Prédiction
        if st.button("🔮 Lancer la Prédiction"):
            with st.spinner("Prédiction en cours..."):
                pred_prob, pred_binary = predict_tumor_evolution(
                    autoencoder, predictor, t1_brain, t2_brain, t1_mask, t2_mask
                )
            
            st.success("✅ Prédiction terminée!")
            
            # Affichage des résultats
            st.markdown("### 📊 Résultats de Prédiction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Visualisation Comparée")
                slice_idx = st.slider("Slice à visualiser", 0, pred_prob.shape[0] - 1, pred_prob.shape[0] // 2)
                comp_img = create_comparison_plot(t1_mask, t2_mask, pred_prob, pred_binary, slice_idx)
                st.image(comp_img, use_column_width=True)
            
            with col2:
                st.subheader("Visualisation 3D")
                st.plotly_chart(create_3d_visualization(pred_prob, "Prédiction Probabiliste T3"))
                st.plotly_chart(create_3d_visualization(pred_binary, "Prédiction Binaire T3"))
            
            # Téléchargements
            st.markdown("### 📥 Télécharger les Résultats")
            
            # Probabiliste
            prob_nii = nib.Nifti1Image(pred_prob, np.eye(4))
            prob_bytes = io.BytesIO()
            nib.save(prob_nii, prob_bytes)
            download_button_with_data(prob_bytes.getvalue(), "pred_prob.nii.gz", "Télécharger Masque Probabiliste")
            
            # Binaire
            bin_nii = nib.Nifti1Image(pred_binary, np.eye(4))
            bin_bytes = io.BytesIO()
            nib.save(bin_nii, bin_bytes)
            download_button_with_data(bin_bytes.getvalue(), "pred_binary.nii.gz", "Télécharger Masque Binaire")

elif selected_tab == "📋 Évaluation":
    st.markdown("## 📋 Évaluation du Modèle")
    
    eval_results = load_evaluation_results()
    
    if eval_results is not None:
        st.dataframe(eval_results)
        
        st.markdown("### 📊 Analyse Globale")
        st.write(eval_results.describe())
        
        st.markdown("### 📈 Visualisation")
        fig = px.box(eval_results, y=['dice', 'iou', 'sensitivity', 'specificity'])
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Aucune évaluation disponible. Lancez evaluate.py pour générer les résultats.")

elif selected_tab == "⚙️ Configuration":
    st.markdown("## ⚙️ Configuration du Système")
    
    config = get_default_config()
    st.json(config)
    
    st.markdown("### 🔧 Personnalisation")
    new_latent_dim = st.number_input("Latent Dim", value=config['latent_dim'])
    if st.button("Appliquer Changements"):
        st.info("Changements appliqués (simulé pour cet exemple).")

elif selected_tab == "📚 Documentation":
    st.markdown("## 📚 Documentation Complète")
    
    st.markdown("""
    ### Introduction
    Cette application permet de prédire l'évolution tumorale à partir d'images IRM.

    ### Utilisation
    1. Chargez les modèles via le dashboard.
    2. Utilisez l'onglet Prédiction pour uploader des fichiers NIfTI.
    3. Analysez les résultats dans les visualisations.

    ### Modèles
    - **AutoEncoder**: Pour l'extraction de features.
    - **ConvLSTM**: Pour la prédiction temporelle.

    Pour plus de détails, consultez le code source.
    """)