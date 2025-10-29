"""Strategy package: holds strategy defaults, implementations, and registry."""

from . import registry

# Import implementations and register them
from . import macd_cross as _macd_cross
from . import macd_pre_cross_below_zero as _macd_pre
from . import macd_convergence_stock as _macd_conv_stock

registry.register('macd_cross', _macd_cross.run)
registry.register('macd_pre_cross_below_zero', _macd_pre.run)
registry.register('macd_convergence_stock', _macd_conv_stock.run)

get_strategy = registry.get
available_strategies = registry.available

