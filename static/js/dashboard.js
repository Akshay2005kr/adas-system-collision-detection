// ADAS Dashboard JavaScript

// Utility functions
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return Number(num).toFixed(decimals);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Auto-refresh for camera status
let autoRefreshInterval = null;

function startAutoRefresh(callback, interval = 1000) {
    stopAutoRefresh();
    callback();
    autoRefreshInterval = setInterval(callback, interval);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// Chart helper
function createGradient(ctx, color) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color.replace(')', ', 0.8)').replace('rgb', 'rgba'));
    gradient.addColorStop(1, color.replace(')', ', 0.1)').replace('rgb', 'rgba'));
    return gradient;
}

// Local storage for preferences
const Preferences = {
    get(key, defaultValue) {
        try {
            const item = localStorage.getItem(`adas_${key}`);
            return item ? JSON.parse(item) : defaultValue;
        } catch {
            return defaultValue;
        }
    },
    set(key, value) {
        try {
            localStorage.setItem(`adas_${key}`, JSON.stringify(value));
        } catch {}
    }
};

// Notification system
function showNotification(message, type = 'info') {
    const container = document.createElement('div');
    container.className = `notification notification-${type}`;
    container.textContent = message;
    container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        background: ${type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(container);
    
    setTimeout(() => {
        container.style.opacity = '0';
        container.style.transition = 'opacity 0.3s';
        setTimeout(() => container.remove(), 300);
    }, 3000);
}

// Add slideIn animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

// File upload drag and drop
function initDragDrop(dropZone, inputElement) {
    if (!dropZone || !inputElement) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            inputElement.files = files;
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            inputElement.dispatchEvent(event);
        }
    }, false);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize drag and drop for video upload
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoFile');
    if (dropZone && videoInput) {
        initDragDrop(dropZone, videoInput);
    }
    
    // Add loading states to forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', () => {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-small"></span> Processing...';
            }
        });
    });
});

// Fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = 30000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for use in other scripts
window.ADASUtils = {
    formatNumber,
    formatTime,
    startAutoRefresh,
    stopAutoRefresh,
    showNotification,
    initDragDrop,
    fetchWithTimeout,
    debounce,
    Preferences
};
