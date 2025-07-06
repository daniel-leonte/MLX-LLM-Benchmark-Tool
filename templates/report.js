/**
 * Benchmark Report Interactive Features
 * Handles model removal and server communication
 */

class BenchmarkReport {
    constructor() {
        this.isServerMode = window.location.protocol === 'http:';
        this.pendingDeletions = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.showServerModeIndicator();
    }

    setupEventListeners() {
        // Hide notice when clicking anywhere else
        document.addEventListener('click', (event) => {
            const removedNotice = document.getElementById('removedNotice');
            if (!removedNotice) return;
            
            const isClickOnNotice = removedNotice.contains(event.target);
            if (!isClickOnNotice && removedNotice.style.display === 'block') {
                setTimeout(() => {
                    removedNotice.style.display = 'none';
                }, 100);
            }
        });
    }

    showServerModeIndicator() {
        if (this.isServerMode) {
            document.addEventListener('DOMContentLoaded', () => {
                const header = document.querySelector('.header');
                if (header) {
                    const serverBadge = document.createElement('div');
                    serverBadge.className = 'server-mode-indicator';
                    serverBadge.innerHTML = '🌐 <strong>Server Mode:</strong> Deletions are permanent and saved to database';
                    header.appendChild(serverBadge);
                }
            });
        }
    }

    removeModel(modelId, modelName, runId) {
        if (this.isServerMode) {
            this.removeModelFromServer(modelId, modelName, runId);
        } else {
            this.removeModelLocally(modelId, modelName, runId);
        }
    }

    removeModelFromServer(modelId, modelName, runId) {
        const modelElement = document.getElementById('model-' + modelId);
        const tableRow = document.getElementById('table-row-' + modelId);
        
        // Show loading state
        if (modelElement) {
            modelElement.style.opacity = '0.5';
        }
        if (tableRow) {
            tableRow.style.opacity = '0.5';
        }

        // Send DELETE request to server
        const url = '/api/remove-result?model_name=' + encodeURIComponent(modelName) + '&run_id=' + encodeURIComponent(runId);
        
        fetch(url, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to remove result from server');
            }
        })
        .then(data => {
            // Success: The server will regenerate the HTML, so we reload the page
            this.showNotification('Model "' + modelName + '" permanently removed from database. Reloading...', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        })
        .catch(error => {
            console.error('Error removing model:', error);
            
            // Restore opacity
            if (modelElement) {
                modelElement.style.opacity = '1';
            }
            if (tableRow) {
                tableRow.style.opacity = '1';
            }
            
            this.showNotification('Error removing model: ' + error.message, 'error');
        });
    }

    removeModelLocally(modelId, modelName, runId) {
        const modelElement = document.getElementById('model-' + modelId);
        const tableRow = document.getElementById('table-row-' + modelId);
        
        // Store for potential undo
        this.lastRemovedElements = { 
            tableRow: tableRow ? tableRow.cloneNode(true) : null,
            modelSection: modelElement ? modelElement.cloneNode(true) : null 
        };
        
        // Add fade-out animation
        if (modelElement) {
            modelElement.classList.add('fade-out');
        }
        if (tableRow) {
            tableRow.classList.add('fade-out');
        }
        
        // Remove elements after animation
        setTimeout(() => {
            if (modelElement) {
                modelElement.remove();
            }
            if (tableRow) {
                tableRow.remove();
            }
            
            this.showNotification('Model "' + modelName + '" removed locally (not from database). <a href="#" onclick="benchmarkReport.undoLocalRemove()">Undo</a>', 'warning');
        }, 300);
    }

    showNotification(message, type = 'info') {
        const removedNotice = document.getElementById('removedNotice');
        if (!removedNotice) return;
        
        removedNotice.className = 'removed-notice ' + type;
        removedNotice.innerHTML = '<strong>' + message + '</strong>';
        removedNotice.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            removedNotice.style.display = 'none';
        }, 10000);
    }

    undoLocalRemove() {
        // This would be more complex to implement properly
        // For now, just suggest reloading the page
        this.showNotification('To restore locally removed items, please reload the page.', 'info');
    }
}

// Initialize the report when DOM is loaded
let benchmarkReport;

document.addEventListener('DOMContentLoaded', () => {
    benchmarkReport = new BenchmarkReport();
});

// Global function for button onclick handlers (for backward compatibility)
function removeModel(modelId, modelName, runId) {
    if (benchmarkReport) {
        benchmarkReport.removeModel(modelId, modelName, runId);
    }
} 