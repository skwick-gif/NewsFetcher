/**
 * Settings Manager Module
 * Handles settings persistence using localStorage
 */

class SettingsManager {
    constructor() {
        this.storageKey = 'tariffRadarSettings';
        this.defaultSettings = {
            updateInterval: 30,
            maxArticles: 50,
            keywordThreshold: 0.3,
            highScoreThreshold: 0.7,
            mediumScoreThreshold: 0.4,
            autoRefresh: true,
            enableNotifications: false,
            enableWebSocket: true,
            darkMode: true,
            compactView: false,
            showCharts: true,
            enableMLPredictions: true,
            debugMode: false
        };
    }

    /**
     * Initialize settings manager
     */
    init() {
        this.loadSettings();
        console.log('✅ Settings Manager initialized');
    }

    /**
     * Load settings from localStorage and apply to UI
     */
    loadSettings() {
        try {
            const savedSettings = localStorage.getItem(this.storageKey);
            
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                
                // Apply each setting to the UI
                Object.keys(settings).forEach(key => {
                    const element = document.getElementById(key);
                    if (element) {
                        if (element.type === 'checkbox') {
                            element.checked = settings[key];
                        } else if (element.type === 'number' || element.type === 'text') {
                            element.value = settings[key];
                        }
                    }
                });
                
                console.log('✅ Settings loaded from localStorage');
            } else {
                // Apply default settings
                this.applyDefaultSettings();
                console.log('ℹ️ Using default settings');
            }
        } catch (error) {
            console.error('❌ Error loading settings:', error);
            this.applyDefaultSettings();
        }
    }

    /**
     * Apply default settings to UI
     */
    applyDefaultSettings() {
        Object.keys(this.defaultSettings).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = this.defaultSettings[key];
                } else if (element.type === 'number' || element.type === 'text') {
                    element.value = this.defaultSettings[key];
                }
            }
        });
    }

    /**
     * Save current settings to localStorage
     */
    saveSettings() {
        try {
            const settings = {
                updateInterval: this.getInputValue('updateInterval', 'number'),
                maxArticles: this.getInputValue('maxArticles', 'number'),
                keywordThreshold: this.getInputValue('keywordThreshold', 'number'),
                highScoreThreshold: this.getInputValue('highScoreThreshold', 'number'),
                mediumScoreThreshold: this.getInputValue('mediumScoreThreshold', 'number'),
                autoRefresh: this.getCheckboxValue('autoRefresh'),
                enableNotifications: this.getCheckboxValue('enableNotifications'),
                enableWebSocket: this.getCheckboxValue('enableWebSocket'),
                darkMode: this.getCheckboxValue('darkMode'),
                compactView: this.getCheckboxValue('compactView'),
                showCharts: this.getCheckboxValue('showCharts'),
                enableMLPredictions: this.getCheckboxValue('enableMLPredictions'),
                debugMode: this.getCheckboxValue('debugMode')
            };

            localStorage.setItem(this.storageKey, JSON.stringify(settings));
            
            // Show success message
            this.showSaveMessage('Settings saved successfully! ✅', 'success');
            
            console.log('✅ Settings saved:', settings);
            
            return settings;
        } catch (error) {
            console.error('❌ Error saving settings:', error);
            this.showSaveMessage('Failed to save settings ❌', 'error');
            return null;
        }
    }

    /**
     * Get input value with fallback to default
     */
    getInputValue(id, type) {
        const element = document.getElementById(id);
        if (!element) return this.defaultSettings[id];
        
        const value = element.value;
        
        if (type === 'number') {
            const parsed = parseFloat(value);
            return isNaN(parsed) ? this.defaultSettings[id] : parsed;
        }
        
        return value || this.defaultSettings[id];
    }

    /**
     * Get checkbox value with fallback to default
     */
    getCheckboxValue(id) {
        const element = document.getElementById(id);
        if (!element) return this.defaultSettings[id];
        return element.checked;
    }

    /**
     * Reset settings to defaults with confirmation
     */
    resetSettings() {
        const confirmed = confirm('Are you sure you want to reset all settings to defaults?');
        
        if (confirmed) {
            try {
                localStorage.removeItem(this.storageKey);
                this.applyDefaultSettings();
                
                this.showSaveMessage('Settings reset to defaults! 🔄', 'success');
                console.log('✅ Settings reset to defaults');
            } catch (error) {
                console.error('❌ Error resetting settings:', error);
                this.showSaveMessage('Failed to reset settings ❌', 'error');
            }
        }
    }

    /**
     * Get current settings object (for use by other modules)
     */
    getSettings() {
        try {
            const savedSettings = localStorage.getItem(this.storageKey);
            if (savedSettings) {
                return JSON.parse(savedSettings);
            }
        } catch (error) {
            console.error('Error reading settings:', error);
        }
        
        return this.defaultSettings;
    }

    /**
     * Get a specific setting value
     */
    getSetting(key) {
        const settings = this.getSettings();
        return settings[key] !== undefined ? settings[key] : this.defaultSettings[key];
    }

    /**
     * Update a specific setting
     */
    setSetting(key, value) {
        const settings = this.getSettings();
        settings[key] = value;
        
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(settings));
            console.log(`✅ Setting updated: ${key} = ${value}`);
            return true;
        } catch (error) {
            console.error(`❌ Error updating setting ${key}:`, error);
            return false;
        }
    }

    /**
     * Show save message to user
     */
    showSaveMessage(message, type) {
        // Try to find a message container
        let messageElement = document.getElementById('settings-message');
        
        if (!messageElement) {
            // Create temporary message element
            messageElement = document.createElement('div');
            messageElement.id = 'settings-message';
            messageElement.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 8px;
                font-weight: bold;
                z-index: 10000;
                animation: fadeIn 0.3s;
                background: ${type === 'success' ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)'};
                color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            `;
            document.body.appendChild(messageElement);
        }
        
        messageElement.textContent = message;
        messageElement.style.display = 'block';
        
        // Hide after 3 seconds
        setTimeout(() => {
            if (messageElement && messageElement.parentNode) {
                messageElement.style.display = 'none';
            }
        }, 3000);
    }

    /**
     * Export settings as JSON file
     */
    exportSettings() {
        const settings = this.getSettings();
        const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `tariff-radar-settings-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        
        console.log('✅ Settings exported');
        this.showSaveMessage('Settings exported! 📥', 'success');
    }

    /**
     * Import settings from JSON file
     */
    importSettings(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            try {
                const settings = JSON.parse(e.target.result);
                localStorage.setItem(this.storageKey, JSON.stringify(settings));
                this.loadSettings();
                
                this.showSaveMessage('Settings imported! 📤', 'success');
                console.log('✅ Settings imported:', settings);
            } catch (error) {
                console.error('❌ Error importing settings:', error);
                this.showSaveMessage('Invalid settings file ❌', 'error');
            }
        };
        
        reader.readAsText(file);
    }
}

// Export for use in main.js
window.SettingsManager = SettingsManager;