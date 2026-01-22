"""Constants and color schemes for the Bus GUI."""

from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from core.constants import Zone, BuildingType


# Player colors
PLAYER_COLORS = [
    QColor("#E63946"),  # Red - Player 0
    QColor("#457B9D"),  # Blue - Player 1
    QColor("#2A9D8F"),  # Teal - Player 2
    QColor("#E9C46A"),  # Yellow - Player 3
    QColor("#9B5DE5"),  # Purple - Player 4
]

PLAYER_COLOR_NAMES = ["Red", "Blue", "Teal", "Yellow", "Purple"]

# Zone colors (for building slot borders)
ZONE_COLORS = {
    Zone.A: QColor("#FF6B6B"),  # Red (innermost)
    Zone.B: QColor("#4ECDC4"),  # Teal
    Zone.C: QColor("#45B7D1"),  # Blue
    Zone.D: QColor("#96CEB4"),  # Green (outermost)
}

# Building colors
BUILDING_COLORS = {
    BuildingType.HOUSE: QColor("#FF9999"),   # Light red
    BuildingType.OFFICE: QColor("#99CCFF"),  # Light blue
    BuildingType.PUB: QColor("#FFCC99"),     # Light orange
}

BUILDING_BORDER_COLORS = {
    BuildingType.HOUSE: QColor("#CC0000"),   # Dark red
    BuildingType.OFFICE: QColor("#0066CC"),  # Dark blue
    BuildingType.PUB: QColor("#CC6600"),     # Dark orange
}

# Node colors
NODE_COLORS = {
    "regular": QColor("#FFFFFF"),       # White for regular nodes
    "train_station": QColor("#FFD700"), # Gold for train stations
    "central_park": QColor("#90EE90"),  # Light green for central parks
}

# Edge colors
EDGE_COLORS = {
    "empty": QColor("#CCCCCC"),         # Gray for empty edges
    "hover": QColor("#FFD700"),         # Gold for hover
}

# Passenger colors
PASSENGER_COLOR = QColor("#8B0000")      # Dark red
PASSENGER_BG_COLOR = QColor("#FFEEEE")   # Light pink background

# UI dimensions
NODE_RADIUS = 25
EDGE_WIDTH = 2
RAIL_WIDTH = 4
BUILDING_SIZE = 16
PASSENGER_BADGE_SIZE = 20

# Window dimensions
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
BOARD_MIN_WIDTH = 800
BOARD_MIN_HEIGHT = 600
ACTION_BOARD_WIDTH = 400
PLAYER_INFO_HEIGHT = 150
GAME_INFO_HEIGHT = 100

# Fonts
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 11
SMALL_FONT_SIZE = 9

# Time clock colors
TIME_CLOCK_COLORS = {
    BuildingType.HOUSE: QColor("#FF9999"),
    BuildingType.OFFICE: QColor("#99CCFF"),
    BuildingType.PUB: QColor("#FFCC99"),
}

# Action area names for display
ACTION_AREA_NAMES = {
    "line_expansion": "Line Expansion",
    "buses": "Buses",
    "passengers": "Passengers",
    "buildings": "Buildings",
    "time_clock": "Time Clock",
    "vrroomm": "Vrroomm!",
    "starting_player": "Starting Player",
}
