// Loader personalizado para Banco de Corrientes
class CorrientesLoader {
    constructor() {
        this.loaderElement = null;
        this.statusElement = null;
        this.progressBar = null;
        this.isVisible = false;
        this.init();
    }

    init() {
        // Crear el elemento del loader
        this.createLoaderElement();
        
        // Mostrar el loader inmediatamente
        this.show();
        
        // Simular progreso de carga
        this.simulateLoading();
    }

    createLoaderElement() {
        // Crear el contenedor principal
        this.loaderElement = document.createElement('div');
        this.loaderElement.className = 'loader-container';
        this.loaderElement.id = 'corrientes-loader';

        // Crear el contenido del loader
        this.loaderElement.innerHTML = `
            <div class="loader-logo">
                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiByeD0iNDAiIGZpbGw9IiMxZTNhOGEiLz4KPHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4PSIxMCIgeT0iMTAiPgo8cGF0aCBkPSJNMTUgMjVIMzBWNDVIMTVWMjVaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMzUgMjVINDBWNDVIMzVWMjVaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNNDUgMjVINjBWNDVINDBWMjVaIiBmaWxsPSJ3aGl0ZSIvPgo8Y2lyY2xlIGN4PSIzMCIgY3k9IjE1IiByPSI1IiBmaWxsPSIjZjU5ZTBhIi8+CjxjaXJjbGUgY3g9IjQ1IiBjeT0iMTUiIHI9IjUiIGZpbGw9IiNmNTllMGEiLz4KPC9zdmc+Cjwvc3ZnPgo=" alt="Banco de Corrientes">
            </div>
            <div class="loader-text">🏦 Banco de Corrientes</div>
            <div class="loader-subtitle">Asistente Inteligente</div>
            <div class="loader-spinner"></div>
            <div class="loader-progress">
                <div class="loader-progress-bar"></div>
            </div>
            <div class="loader-status" id="loader-status">Inicializando sistema...</div>
        `;

        // Agregar al body
        document.body.appendChild(this.loaderElement);
        
        // Obtener referencias a elementos
        this.statusElement = document.getElementById('loader-status');
        this.progressBar = this.loaderElement.querySelector('.loader-progress-bar');
    }

    show() {
        if (this.loaderElement) {
            this.loaderElement.style.display = 'flex';
            this.isVisible = true;
        }
    }

    hide() {
        if (this.loaderElement) {
            this.loaderElement.classList.add('loader-fade-out');
            setTimeout(() => {
                this.loaderElement.style.display = 'none';
                this.isVisible = false;
            }, 500);
        }
    }

    updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
    }

    simulateLoading() {
        const loadingSteps = [
            { message: "Inicializando sistema...", delay: 500 },
            { message: "Cargando modelos de IA...", delay: 1000 },
            { message: "Conectando con Ollama...", delay: 800 },
            { message: "Configurando Mistral...", delay: 1200 },
            { message: "Preparando interfaz...", delay: 600 },
            { message: "¡Listo!", delay: 300 }
        ];

        let currentStep = 0;

        const showNextStep = () => {
            if (currentStep < loadingSteps.length) {
                const step = loadingSteps[currentStep];
                this.updateStatus(step.message);
                currentStep++;
                
                setTimeout(showNextStep, step.delay);
            } else {
                // Ocultar el loader después de completar todos los pasos
                setTimeout(() => {
                    this.hide();
                }, 500);
            }
        };

        showNextStep();
    }

    // Método para mostrar el loader manualmente
    showWithMessage(message) {
        this.show();
        this.updateStatus(message);
    }

    // Método para ocultar el loader manualmente
    hideLoader() {
        this.hide();
    }
}

// Inicializar el loader cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    window.corrientesLoader = new CorrientesLoader();
});

// Función global para mostrar/ocultar el loader desde Streamlit
window.showCorrientesLoader = function(message = "Cargando...") {
    if (window.corrientesLoader) {
        window.corrientesLoader.showWithMessage(message);
    }
};

window.hideCorrientesLoader = function() {
    if (window.corrientesLoader) {
        window.corrientesLoader.hideLoader();
    }
}; 