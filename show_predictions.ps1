# Progressive ML Predictions - Pretty Display Script
# Save as: show_predictions.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$Symbol = "AAPL",
    
    [Parameter(Mandatory=$false)]
    [string]$Mode = "progressive"
)

$ErrorActionPreference = "Stop"

try {
    # Get predictions from API
    Write-Host "`n🔄 Getting predictions for $Symbol..." -ForegroundColor Cyan
    $result = Invoke-RestMethod -Uri "http://localhost:8000/api/ml/progressive/predict/$Symbol?mode=$Mode" -Method POST
    
    if ($result.status -ne 'success') {
        Write-Host "❌ Failed to get predictions" -ForegroundColor Red
        exit 1
    }
    
    $pred = $result.predictions
    
    # Header
    Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  📊 Progressive ML Predictions for $($pred.symbol.PadRight(28)) ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    # Current Price
    Write-Host "`n💵 Current Price: " -NoNewline -ForegroundColor Yellow
    Write-Host "`$$([math]::Round($pred.current_price, 2))" -ForegroundColor White
    Write-Host "📅 As of: $($pred.current_date)" -ForegroundColor Gray
    
    # Predictions for each horizon
    Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                      📈 PREDICTIONS                           ║" -ForegroundColor Green
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    
    foreach($horizonKey in ($pred.predictions.PSObject.Properties.Name | Sort-Object)) {
        $p = $pred.predictions.$horizonKey
        
        $priceChangeColor = if($p.price_change_pct -gt 0.02) { 'Green' } 
                           elseif($p.price_change_pct -lt -0.02) { 'Red' } 
                           else { 'Yellow' }
        
        $signalColor = switch($p.signal) {
            'BUY' { 'Green' }
            'SELL' { 'Red' }
            default { 'Yellow' }
        }
        
        Write-Host "`n┌─────────────────────────────────────────────────────────────┐" -ForegroundColor White
        Write-Host "│  ⏰ Time Horizon: " -NoNewline -ForegroundColor White
        Write-Host "$horizonKey".PadRight(44) -NoNewline -ForegroundColor Cyan
        Write-Host "│" -ForegroundColor White
        Write-Host "├─────────────────────────────────────────────────────────────┤" -ForegroundColor White
        
        # Target Price
        Write-Host "│  🎯 Target Price:      " -NoNewline -ForegroundColor White
        Write-Host "`$$([math]::Round($p.target_price, 2))".PadRight(35) -NoNewline -ForegroundColor Cyan
        Write-Host "│" -ForegroundColor White
        
        # Price Change
        $changeStr = "$([math]::Round($p.price_change_pct * 100, 2))%"
        $changeAbs = "$([math]::Round($p.price_change_abs, 2))"
        Write-Host "│  📊 Expected Change:   " -NoNewline -ForegroundColor White
        Write-Host "$changeStr ($changeAbs)".PadRight(35) -NoNewline -ForegroundColor $priceChangeColor
        Write-Host "│" -ForegroundColor White
        
        # Confidence
        $confStr = "$([math]::Round($p.confidence * 100, 1))%"
        Write-Host "│  🎲 Confidence Level:  " -NoNewline -ForegroundColor White
        Write-Host $confStr.PadRight(35) -NoNewline -ForegroundColor Yellow
        Write-Host "│" -ForegroundColor White
        
        # Trading Signal
        Write-Host "│  📡 Trading Signal:    " -NoNewline -ForegroundColor White
        Write-Host "$($p.signal)".PadRight(35) -NoNewline -ForegroundColor $signalColor
        Write-Host "│" -ForegroundColor White
        
        # Direction
        $dirStr = "$($p.direction) ($([math]::Round($p.direction_prob * 100, 1))%)"
        Write-Host "│  🔼 Direction:         " -NoNewline -ForegroundColor White
        Write-Host $dirStr.PadRight(35) -NoNewline -ForegroundColor White
        Write-Host "│" -ForegroundColor White
        
        # Number of models
        Write-Host "│  🤖 Models Used:       " -NoNewline -ForegroundColor White
        Write-Host "$($p.num_models)".PadRight(35) -NoNewline -ForegroundColor Gray
        Write-Host "│" -ForegroundColor White
        
        Write-Host "└─────────────────────────────────────────────────────────────┘" -ForegroundColor White
    }
    
    # Overall Sentiment
    Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "║                   🎭 OVERALL SENTIMENT                        ║" -ForegroundColor Magenta
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    
    $sentiment = $pred.overall_sentiment
    $sentimentColor = switch($sentiment.sentiment) {
        'BULLISH' { 'Green' }
        'BEARISH' { 'Red' }
        default { 'Yellow' }
    }
    
    Write-Host "`n  Sentiment:     " -NoNewline -ForegroundColor White
    Write-Host $sentiment.sentiment -ForegroundColor $sentimentColor
    Write-Host "  Strength:      " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($sentiment.strength * 100, 2))%" -ForegroundColor Yellow
    Write-Host "  Avg Change:    " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($sentiment.avg_price_change * 100, 2))%" -ForegroundColor Cyan
    Write-Host "  Confidence:    " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($sentiment.confidence * 100, 1))%" -ForegroundColor Yellow
    
    # Risk Metrics
    Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║                     ⚠️  RISK ANALYSIS                         ║" -ForegroundColor Red
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    
    $risk = $pred.risk_metrics
    $riskColor = switch($risk.risk_level) {
        'HIGH' { 'Red' }
        'MEDIUM' { 'Yellow' }
        default { 'Green' }
    }
    
    Write-Host "`n  Risk Level:         " -NoNewline -ForegroundColor White
    Write-Host $risk.risk_level -ForegroundColor $riskColor
    Write-Host "  Volatility:         " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($risk.volatility * 100, 2))%" -ForegroundColor Yellow
    Write-Host "  Max Potential Gain: " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($risk.max_gain * 100, 2))%" -ForegroundColor Green
    Write-Host "  Max Potential Loss: " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($risk.max_loss * 100, 2))%" -ForegroundColor Red
    Write-Host "  Risk/Reward Ratio:  " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($risk.risk_reward_ratio, 2))" -ForegroundColor Cyan
    Write-Host "  Price Range:        " -NoNewline -ForegroundColor White
    Write-Host "`$$([math]::Round($risk.prediction_range, 2))" -ForegroundColor Yellow
    
    Write-Host "`n═══════════════════════════════════════════════════════════════`n" -ForegroundColor Gray
    
} catch {
    Write-Host "`n❌ Error: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
