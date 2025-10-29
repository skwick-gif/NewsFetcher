from flask import Blueprint, render_template

# UI blueprint: serve template pages only (no business logic)
ui_bp = Blueprint('ui', __name__)

@ui_bp.route('/docs/progressive-ml')
def docs_progressive_ml():
    """Serve the Progressive ML Guide (static HTML) from templates/docs.
    This route was extracted from app/server.py to reduce monolith size.
    """
    return render_template('docs/progressive_ml_guide.html')


# Additional UI routes extracted from app/server.py

@ui_bp.route('/')
def dashboard():
    """Main MarketPulse dashboard (legacy kept for compatibility)."""
    return render_template('dashboard.html')


@ui_bp.route('/rl')
def rl_dashboard():
    """RL Dashboard page (experimental)."""
    return render_template('rl_dashboard.html')


@ui_bp.route('/scanner')
def scanner_page():
    """Scanner page (modern scanner UI)."""
    return render_template('scanner/scanner.html')


@ui_bp.route('/strategy')
def strategy_lab_page():
    """Strategy Lab page."""
    return render_template('strategy/lab.html')
