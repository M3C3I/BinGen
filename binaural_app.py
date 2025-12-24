#!/usr/bin/env python3
"""
Binaural Audio Generator Application
A comprehensive tool for creating customizable binaural audio experiences.

Features:
- Multiple independent audio sources with stereo output
- Time-based parameter sequences with smooth transitions
- Source grouping with shared/independent editing
- Interactive visualization of frequency and volume curves
- Real-time low-latency audio playback
- Project save/load functionality
- Undo/redo support
- Modern, intuitive interface
"""

import sys
import json
import copy
import uuid
import math
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from enum import Enum
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QSlider,
    QGroupBox, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QDialogButtonBox, QMessageBox, QFileDialog,
    QTabWidget, QCheckBox, QMenu, QToolBar, QStatusBar, QProgressBar,
    QFrame, QSizePolicy, QStyle, QStyleFactory, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, Slot
from PySide6.QtGui import (
    QAction, QKeySequence, QIcon, QColor, QPalette, QFont,
    QShortcut, QPen, QBrush
)

import pyqtgraph as pg
import sounddevice as sd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TransitionStyle(Enum):
    """Interpolation curve types for parameter transitions."""
    LINEAR = "Linear"
    EASE_IN = "Ease In (Quadratic)"
    EASE_OUT = "Ease Out (Quadratic)"
    EASE_IN_OUT = "Ease In/Out (Quadratic)"
    EASE_IN_CUBIC = "Ease In (Cubic)"
    EASE_OUT_CUBIC = "Ease Out (Cubic)"
    EASE_IN_OUT_CUBIC = "Ease In/Out (Cubic)"
    EXPONENTIAL_IN = "Exponential In"
    EXPONENTIAL_OUT = "Exponential Out"
    SINE_IN = "Sine In"
    SINE_OUT = "Sine Out"
    SINE_IN_OUT = "Sine In/Out"
    STEP = "Step (Instant)"


class WaveformType(Enum):
    """Available waveform types for audio generation."""
    SINE = "Sine"
    TRIANGLE = "Triangle"
    SAWTOOTH = "Sawtooth"
    SQUARE = "Square"
    SOFT_SQUARE = "Soft Square"


# =============================================================================
# Configuration Constants
# =============================================================================

# Audio settings
DEFAULT_SAMPLE_RATE = 44100  # CD-quality sample rate (Hz)
DEFAULT_BUFFER_SIZE = 1024   # Audio buffer size in samples (affects latency)
MIN_FREQUENCY = 20.0         # Lower bound of human hearing (Hz)
MAX_FREQUENCY = 2000.0       # Upper limit for binaural tones (Hz) - higher causes aliasing issues
DEFAULT_FREQUENCY = 440.0    # A4 concert pitch (Hz)

# Binaural settings
BINAURAL_BEAT_OFFSET = 10.0  # Default L/R frequency difference (Hz) - creates 10Hz alpha wave beat

# Duration limits
DEFAULT_DURATION = 60.0      # Default project duration (seconds)
MIN_EVENT_DURATION = 0.01    # Minimum event duration to prevent division by zero (seconds)

# Waveform parameters
SOFT_SQUARE_SHARPNESS = 4    # tanh coefficient for soft square wave (higher = sharper edges)

# Visualization
VIZ_POINTS_PER_SECOND = 10   # Resolution for timeline visualization
VIZ_MAX_POINTS = 2000        # Maximum points to prevent performance issues
VIZ_MIN_POINTS = 100         # Minimum points for short durations

# =============================================================================
# Easing Functions
# =============================================================================

def ease_linear(t: float) -> float:
    return t

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return 1 - (1 - t) * (1 - t)

def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

def ease_in_cubic(t: float) -> float:
    return t * t * t

def ease_out_cubic(t: float) -> float:
    return 1 - pow(1 - t, 3)

def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def ease_exponential_in(t: float) -> float:
    return 0 if t == 0 else pow(2, 10 * t - 10)

def ease_exponential_out(t: float) -> float:
    return 1 if t == 1 else 1 - pow(2, -10 * t)

def ease_sine_in(t: float) -> float:
    return 1 - math.cos((t * math.pi) / 2)

def ease_sine_out(t: float) -> float:
    return math.sin((t * math.pi) / 2)

def ease_sine_in_out(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2

def ease_step(t: float) -> float:
    return 0 if t < 1 else 1


EASING_FUNCTIONS: dict[TransitionStyle, Callable[[float], float]] = {
    TransitionStyle.LINEAR: ease_linear,
    TransitionStyle.EASE_IN: ease_in_quad,
    TransitionStyle.EASE_OUT: ease_out_quad,
    TransitionStyle.EASE_IN_OUT: ease_in_out_quad,
    TransitionStyle.EASE_IN_CUBIC: ease_in_cubic,
    TransitionStyle.EASE_OUT_CUBIC: ease_out_cubic,
    TransitionStyle.EASE_IN_OUT_CUBIC: ease_in_out_cubic,
    TransitionStyle.EXPONENTIAL_IN: ease_exponential_in,
    TransitionStyle.EXPONENTIAL_OUT: ease_exponential_out,
    TransitionStyle.SINE_IN: ease_sine_in,
    TransitionStyle.SINE_OUT: ease_sine_out,
    TransitionStyle.SINE_IN_OUT: ease_sine_in_out,
    TransitionStyle.STEP: ease_step,
}


def get_easing_function(style: TransitionStyle) -> Callable[[float], float]:
    """Get easing function with fallback to linear if style is invalid."""
    return EASING_FUNCTIONS.get(style, ease_linear)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ParameterEvent:
    """A single timed event specifying target parameters."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = 0.0  # seconds
    duration: float = 5.0  # seconds
    left_frequency: float = DEFAULT_FREQUENCY
    right_frequency: float = DEFAULT_FREQUENCY + BINAURAL_BEAT_OFFSET
    left_volume: float = 0.7
    right_volume: float = 0.7
    
    def __post_init__(self):
        """Validate and sanitize values after initialization."""
        self.validate()
    
    def validate(self):
        """Ensure all values are within valid ranges."""
        # Ensure non-negative start time
        self.start_time = max(0.0, float(self.start_time))
        # Ensure minimum duration to prevent division by zero
        self.duration = max(MIN_EVENT_DURATION, float(self.duration))
        # Clamp frequencies to valid range
        self.left_frequency = max(MIN_FREQUENCY, min(MAX_FREQUENCY, float(self.left_frequency)))
        self.right_frequency = max(MIN_FREQUENCY, min(MAX_FREQUENCY, float(self.right_frequency)))
        # Clamp volumes to 0-1 range
        self.left_volume = max(0.0, min(1.0, float(self.left_volume)))
        self.right_volume = max(0.0, min(1.0, float(self.right_volume)))
    
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'start_time': self.start_time,
            'duration': self.duration,
            'left_frequency': self.left_frequency,
            'right_frequency': self.right_frequency,
            'left_volume': self.left_volume,
            'right_volume': self.right_volume,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ParameterEvent':
        """Create ParameterEvent from dict with safe defaults for missing keys."""
        try:
            return cls(
                id=data.get('id', str(uuid.uuid4())),
                start_time=data.get('start_time', 0.0),
                duration=data.get('duration', 5.0),
                left_frequency=data.get('left_frequency', DEFAULT_FREQUENCY),
                right_frequency=data.get('right_frequency', DEFAULT_FREQUENCY + BINAURAL_BEAT_OFFSET),
                left_volume=data.get('left_volume', 0.7),
                right_volume=data.get('right_volume', 0.7),
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Error parsing ParameterEvent, using defaults: {e}")
            return cls()


@dataclass
class AudioSource:
    """An independent audio source with its own parameter sequence."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Source"
    enabled: bool = True
    transition_style: TransitionStyle = TransitionStyle.EASE_IN_OUT
    waveform: WaveformType = WaveformType.SINE
    events: list[ParameterEvent] = field(default_factory=list)
    group_id: Optional[str] = None
    color: str = "#3498db"  # For visualization
    
    # Runtime state (not persisted)
    _phase_left: float = field(default=0.0, repr=False)
    _phase_right: float = field(default=0.0, repr=False)
    _sorted_events_cache: Optional[list[ParameterEvent]] = field(default=None, repr=False)
    _events_hash: int = field(default=0, repr=False)
    
    def __post_init__(self):
        if not self.events:
            # Add a default event
            self.events.append(ParameterEvent())
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate the sorted events cache."""
        self._sorted_events_cache = None
        self._events_hash = 0
    
    def _get_events_hash(self) -> int:
        """Get a hash of current events for cache validation."""
        event_tuples = tuple((e.id, e.start_time, e.duration) for e in self.events)
        return hash((len(self.events), event_tuples))
    
    def reset_phase(self):
        self._phase_left = 0.0
        self._phase_right = 0.0
    
    def get_sorted_events(self) -> list[ParameterEvent]:
        """Get events sorted by start time (cached for performance)."""
        current_hash = self._get_events_hash()
        if self._sorted_events_cache is None or self._events_hash != current_hash:
            self._sorted_events_cache = sorted(self.events, key=lambda e: e.start_time)
            self._events_hash = current_hash
        return self._sorted_events_cache
    
    def mark_events_changed(self):
        """Call this after modifying events to invalidate cache."""
        self._invalidate_cache()
    
    def create_playback_snapshot(self) -> 'AudioSourceSnapshot':
        """Create a thread-safe snapshot for audio playback."""
        return AudioSourceSnapshot(
            enabled=self.enabled,
            transition_style=self.transition_style,
            waveform=self.waveform,
            events=copy.deepcopy(self.get_sorted_events()),
            phase_left=self._phase_left,
            phase_right=self._phase_right,
        )
    
    def check_overlaps(self) -> list[tuple[ParameterEvent, ParameterEvent]]:
        """Check for overlapping events. Returns list of overlapping pairs."""
        overlaps = []
        sorted_events = self.get_sorted_events()
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            if current.end_time() > next_event.start_time:
                overlaps.append((current, next_event))
        return overlaps
    
    def has_overlaps(self) -> bool:
        """Check if any events overlap."""
        return len(self.check_overlaps()) > 0
    
    def get_parameters_at_time(self, t: float) -> tuple[float, float, float, float]:
        """
        Get interpolated parameters at time t.
        Returns: (left_freq, right_freq, left_vol, right_vol)
        Handles overlapping events by blending them together.
        """
        if not self.events:
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        sorted_events = self.get_sorted_events()
        
        # Before first event
        if t < sorted_events[0].start_time:
            e = sorted_events[0]
            return (e.left_frequency, e.right_frequency, 0.0, 0.0)
        
        # After last event - return silence
        last = sorted_events[-1]
        if t >= last.end_time():
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        # Find all events that contain this time point (handles overlaps)
        active_events = []
        for i, event in enumerate(sorted_events):
            if event.start_time <= t < event.end_time():
                active_events.append((i, event))
        
        # Also check for gaps - find the most recent event if in a gap
        if not active_events:
            for i, event in enumerate(sorted_events):
                if i < len(sorted_events) - 1:
                    next_event = sorted_events[i + 1]
                    if event.end_time() <= t < next_event.start_time:
                        # In gap - hold previous values
                        return (event.left_frequency, event.right_frequency,
                                event.left_volume, event.right_volume)
            # Fallback
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        # Get easing function with fallback
        easing_func = get_easing_function(self.transition_style)
        
        # If only one active event, use standard interpolation
        if len(active_events) == 1:
            i, event = active_events[0]
            # Prevent division by zero (should be caught by validation, but be safe)
            duration = max(event.duration, MIN_EVENT_DURATION)
            progress = (t - event.start_time) / duration
            progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
            eased = easing_func(progress)
            
            if i == 0:
                # First event - fade in from silence
                return (
                    event.left_frequency,
                    event.right_frequency,
                    event.left_volume * eased,
                    event.right_volume * eased
                )
            else:
                # Transition from previous event
                prev = sorted_events[i - 1]
                return (
                    prev.left_frequency + (event.left_frequency - prev.left_frequency) * eased,
                    prev.right_frequency + (event.right_frequency - prev.right_frequency) * eased,
                    prev.left_volume + (event.left_volume - prev.left_volume) * eased,
                    prev.right_volume + (event.right_volume - prev.right_volume) * eased
                )
        
        # Multiple overlapping events - blend them
        total_lf, total_rf, total_lv, total_rv = 0.0, 0.0, 0.0, 0.0
        total_weight = 0.0
        
        for i, event in active_events:
            duration = max(event.duration, MIN_EVENT_DURATION)
            progress = (t - event.start_time) / duration
            progress = max(0.0, min(1.0, progress))
            eased = easing_func(progress)
            
            # Weight by how far into the event we are (favor later events more as they progress)
            weight = max(eased, 0.001)  # Prevent zero weight
            
            if i == 0:
                lf, rf = event.left_frequency, event.right_frequency
                lv, rv = event.left_volume * eased, event.right_volume * eased
            else:
                prev = sorted_events[i - 1]
                lf = prev.left_frequency + (event.left_frequency - prev.left_frequency) * eased
                rf = prev.right_frequency + (event.right_frequency - prev.right_frequency) * eased
                lv = prev.left_volume + (event.left_volume - prev.left_volume) * eased
                rv = prev.right_volume + (event.right_volume - prev.right_volume) * eased
            
            total_lf += lf * weight
            total_rf += rf * weight
            total_lv += lv * weight
            total_rv += rv * weight
            total_weight += weight
        
        if total_weight > 0:
            return (total_lf / total_weight, total_rf / total_weight,
                    total_lv / total_weight, total_rv / total_weight)
        
        # Fallback
        return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'enabled': self.enabled,
            'transition_style': self.transition_style.name,
            'waveform': self.waveform.name,
            'events': [e.to_dict() for e in self.events],
            'group_id': self.group_id,
            'color': self.color,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AudioSource':
        """Create AudioSource from dict with robust error handling."""
        try:
            events = [ParameterEvent.from_dict(e) for e in data.get('events', [])]
            
            # Parse transition style with fallback
            transition_style = TransitionStyle.EASE_IN_OUT
            if 'transition_style' in data:
                try:
                    transition_style = TransitionStyle[data['transition_style']]
                except KeyError:
                    logger.warning(f"Unknown transition style '{data['transition_style']}', using default")
            
            # Parse waveform with fallback
            waveform = WaveformType.SINE
            if 'waveform' in data:
                try:
                    waveform = WaveformType[data['waveform']]
                except KeyError:
                    logger.warning(f"Unknown waveform '{data['waveform']}', using default")
            
            return cls(
                id=data.get('id', str(uuid.uuid4())),
                name=data.get('name', 'Unnamed Source'),
                enabled=data.get('enabled', True),
                transition_style=transition_style,
                waveform=waveform,
                events=events,
                group_id=data.get('group_id'),
                color=data.get('color', '#3498db'),
            )
        except Exception as e:
            logger.error(f"Error parsing AudioSource: {e}")
            return cls(name="Error Loading Source")

@dataclass
class AudioSourceSnapshot:
    """Thread-safe snapshot of AudioSource for playback."""
    enabled: bool
    transition_style: TransitionStyle
    waveform: WaveformType
    events: list[ParameterEvent]
    phase_left: float = 0.0
    phase_right: float = 0.0
    
    def get_parameters_at_time(self, t: float) -> tuple[float, float, float, float]:
        """Get interpolated parameters at time t (same logic as AudioSource)."""
        if not self.events:
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        sorted_events = self.events  # Already sorted in snapshot
        
        if t < sorted_events[0].start_time:
            e = sorted_events[0]
            return (e.left_frequency, e.right_frequency, 0.0, 0.0)
        
        last = sorted_events[-1]
        if t >= last.end_time():
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        active_events = []
        for i, event in enumerate(sorted_events):
            if event.start_time <= t < event.end_time():
                active_events.append((i, event))
        
        if not active_events:
            for i, event in enumerate(sorted_events):
                if i < len(sorted_events) - 1:
                    next_event = sorted_events[i + 1]
                    if event.end_time() <= t < next_event.start_time:
                        return (event.left_frequency, event.right_frequency,
                                event.left_volume, event.right_volume)
            return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)
        
        easing_func = get_easing_function(self.transition_style)
        
        if len(active_events) == 1:
            i, event = active_events[0]
            duration = max(event.duration, MIN_EVENT_DURATION)
            progress = max(0.0, min(1.0, (t - event.start_time) / duration))
            eased = easing_func(progress)
            
            if i == 0:
                return (event.left_frequency, event.right_frequency,
                        event.left_volume * eased, event.right_volume * eased)
            else:
                prev = sorted_events[i - 1]
                return (
                    prev.left_frequency + (event.left_frequency - prev.left_frequency) * eased,
                    prev.right_frequency + (event.right_frequency - prev.right_frequency) * eased,
                    prev.left_volume + (event.left_volume - prev.left_volume) * eased,
                    prev.right_volume + (event.right_volume - prev.right_volume) * eased
                )
        
        # Blend multiple events
        total_lf, total_rf, total_lv, total_rv, total_weight = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, event in active_events:
            duration = max(event.duration, MIN_EVENT_DURATION)
            progress = max(0.0, min(1.0, (t - event.start_time) / duration))
            eased = easing_func(progress)
            weight = max(eased, 0.001)
            
            if i == 0:
                lf, rf = event.left_frequency, event.right_frequency
                lv, rv = event.left_volume * eased, event.right_volume * eased
            else:
                prev = sorted_events[i - 1]
                lf = prev.left_frequency + (event.left_frequency - prev.left_frequency) * eased
                rf = prev.right_frequency + (event.right_frequency - prev.right_frequency) * eased
                lv = prev.left_volume + (event.left_volume - prev.left_volume) * eased
                rv = prev.right_volume + (event.right_volume - prev.right_volume) * eased
            
            total_lf += lf * weight
            total_rf += rf * weight
            total_lv += lv * weight
            total_rv += rv * weight
            total_weight += weight
        
        if total_weight > 0:
            return (total_lf / total_weight, total_rf / total_weight,
                    total_lv / total_weight, total_rv / total_weight)
        return (DEFAULT_FREQUENCY, DEFAULT_FREQUENCY, 0.0, 0.0)

@dataclass 
class SourceGroup:
    """A group of sources that share a timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Group"
    sync_edits: bool = True  # Whether edits apply to all members
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'sync_edits': self.sync_edits,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SourceGroup':
        """Create SourceGroup from dict with safe defaults."""
        try:
            return cls(
                id=data.get('id', str(uuid.uuid4())),
                name=data.get('name', 'Unnamed Group'),
                sync_edits=data.get('sync_edits', True),
            )
        except Exception as e:
            logger.warning(f"Error parsing SourceGroup: {e}")
            return cls()


@dataclass
class GlobalSettings:
    """Global playback and audio settings."""
    duration: float = DEFAULT_DURATION
    master_volume_left: float = 1.0
    master_volume_right: float = 1.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    buffer_size: int = DEFAULT_BUFFER_SIZE
    limiter_enabled: bool = True
    limiter_threshold: float = 0.95
    
    def __post_init__(self):
        """Validate settings after initialization."""
        self.validate()
    
    def validate(self):
        """Ensure all values are within valid ranges."""
        self.duration = max(1.0, float(self.duration))
        self.master_volume_left = max(0.0, min(2.0, float(self.master_volume_left)))
        self.master_volume_right = max(0.0, min(2.0, float(self.master_volume_right)))
        self.sample_rate = max(8000, min(192000, int(self.sample_rate)))
        self.buffer_size = max(64, min(8192, int(self.buffer_size)))
        self.limiter_threshold = max(0.1, min(1.0, float(self.limiter_threshold)))
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GlobalSettings':
        """Create GlobalSettings from dict with safe defaults for missing keys."""
        try:
            return cls(
                duration=data.get('duration', DEFAULT_DURATION),
                master_volume_left=data.get('master_volume_left', 1.0),
                master_volume_right=data.get('master_volume_right', 1.0),
                sample_rate=data.get('sample_rate', DEFAULT_SAMPLE_RATE),
                buffer_size=data.get('buffer_size', DEFAULT_BUFFER_SIZE),
                limiter_enabled=data.get('limiter_enabled', True),
                limiter_threshold=data.get('limiter_threshold', 0.95),
            )
        except Exception as e:
            logger.warning(f"Error parsing GlobalSettings: {e}")
            return cls()


@dataclass
class Project:
    """Complete project state."""
    name: str = "Untitled Project"
    sources: list[AudioSource] = field(default_factory=list)
    groups: list[SourceGroup] = field(default_factory=list)
    settings: GlobalSettings = field(default_factory=GlobalSettings)
    
    # Lock for thread-safe access during playback
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def get_group(self, group_id: str) -> Optional[SourceGroup]:
        for g in self.groups:
            if g.id == group_id:
                return g
        return None
    
    def get_sources_in_group(self, group_id: str) -> list[AudioSource]:
        return [s for s in self.sources if s.group_id == group_id]
    
    def create_playback_snapshot(self) -> tuple[list['AudioSourceSnapshot'], 'GlobalSettings']:
        """Create thread-safe snapshot of sources and settings for audio playback."""
        with self._lock:
            snapshots = [s.create_playback_snapshot() for s in self.sources if s.enabled]
            settings_copy = GlobalSettings(
                duration=self.settings.duration,
                master_volume_left=self.settings.master_volume_left,
                master_volume_right=self.settings.master_volume_right,
                sample_rate=self.settings.sample_rate,
                buffer_size=self.settings.buffer_size,
                limiter_enabled=self.settings.limiter_enabled,
                limiter_threshold=self.settings.limiter_threshold,
            )
            return snapshots, settings_copy
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'sources': [s.to_dict() for s in self.sources],
            'groups': [g.to_dict() for g in self.groups],
            'settings': self.settings.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Project':
        """Create Project from dict with robust error handling."""
        try:
            sources = []
            for s in data.get('sources', []):
                try:
                    sources.append(AudioSource.from_dict(s))
                except Exception as e:
                    logger.error(f"Error loading source: {e}")
            
            groups = []
            for g in data.get('groups', []):
                try:
                    groups.append(SourceGroup.from_dict(g))
                except Exception as e:
                    logger.error(f"Error loading group: {e}")
            
            settings = GlobalSettings.from_dict(data.get('settings', {}))
            
            return cls(
                name=data.get('name', 'Untitled'),
                sources=sources,
                groups=groups,
                settings=settings,
            )
        except Exception as e:
            logger.error(f"Critical error loading project: {e}")
            return cls(name="Error Loading Project")


# =============================================================================
# Undo/Redo System
# =============================================================================

class UndoStack:
    """Simple undo/redo stack for project states."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
    
    def push(self, state: dict):
        self.undo_stack.append(copy.deepcopy(state))
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
    
    def undo(self, current_state: dict) -> Optional[dict]:
        if not self.undo_stack:
            return None
        self.redo_stack.append(copy.deepcopy(current_state))
        return self.undo_stack.pop()
    
    def redo(self, current_state: dict) -> Optional[dict]:
        if not self.redo_stack:
            return None
        self.undo_stack.append(copy.deepcopy(current_state))
        return self.redo_stack.pop()
    
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0
    
    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


# =============================================================================
# Audio Engine
# =============================================================================

class AudioEngine(QObject):
    """Real-time audio generation and playback engine."""
    
    position_changed = Signal(float)
    playback_finished = Signal()
    playback_error = Signal(str)
    
    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.playing = False
        self.position = 0.0
        self.stream: Optional[sd.OutputStream] = None
        
        # Thread-safe playback state
        self._playback_snapshots: list[AudioSourceSnapshot] = []
        self._playback_settings: Optional[GlobalSettings] = None
        self._playback_lock = threading.Lock()
        
        # Position update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._emit_position)
    
    def generate_waveform(self, phase: np.ndarray, waveform: WaveformType) -> np.ndarray:
        """Generate waveform samples from phase values."""
        if waveform == WaveformType.SINE:
            return np.sin(phase)
        elif waveform == WaveformType.TRIANGLE:
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif waveform == WaveformType.SAWTOOTH:
            return 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        elif waveform == WaveformType.SQUARE:
            return np.sign(np.sin(phase))
        elif waveform == WaveformType.SOFT_SQUARE:
            return np.tanh(SOFT_SQUARE_SHARPNESS * np.sin(phase))
        return np.sin(phase)
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, 
                        time_info, status):
        """Audio stream callback - generates audio in real-time using thread-safe snapshots."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            with self._playback_lock:
                snapshots = self._playback_snapshots
                settings = self._playback_settings
            
            if not settings:
                outdata.fill(0)
                return
            
            sample_rate = settings.sample_rate
            
            # Time array for this chunk
            t = np.linspace(
                self.position,
                self.position + frames / sample_rate,
                frames,
                endpoint=False
            )
            
            # Initialize output
            left_channel = np.zeros(frames)
            right_channel = np.zeros(frames)
            
            # Use snapshot count for normalization
            enabled_sources = len(snapshots)
            
            dt = 1.0 / sample_rate
            
            for snapshot in snapshots:
                if not snapshot.enabled:
                    continue
                
                # Get parameters for all frames (still looped, but necessary for complex logic)
                lf = np.zeros(frames)
                rf = np.zeros(frames)
                lv = np.zeros(frames)
                rv = np.zeros(frames)
                
                for i, time_point in enumerate(t):
                    if time_point > settings.duration:
                        break
                    lf[i], rf[i], lv[i], rv[i] = snapshot.get_parameters_at_time(time_point)
                
                # Vectorize phase accumulation and waveform generation
                phase_increments_left = 2 * np.pi * lf * dt
                phase_increments_right = 2 * np.pi * rf * dt
                
                phases_left = (snapshot.phase_left + np.cumsum(phase_increments_left)) % (2 * np.pi)
                phases_right = (snapshot.phase_right + np.cumsum(phase_increments_right)) % (2 * np.pi)
                
                source_left = self.generate_waveform(phases_left, snapshot.waveform) * lv
                source_right = self.generate_waveform(phases_right, snapshot.waveform) * rv
                
                left_channel += source_left
                right_channel += source_right
                
                # Update snapshot phases for continuity (add total increment)
                snapshot.phase_left = (snapshot.phase_left + np.sum(phase_increments_left)) % (2 * np.pi)
                snapshot.phase_right = (snapshot.phase_right + np.sum(phase_increments_right)) % (2 * np.pi)
            
            # Normalize based on enabled sources count
            if enabled_sources > 1:
                normalization = 1.0 / np.sqrt(enabled_sources)
                left_channel *= normalization
                right_channel *= normalization
            
            # Apply master volume
            left_channel *= settings.master_volume_left
            right_channel *= settings.master_volume_right
            
            # Apply limiter if enabled
            if settings.limiter_enabled:
                threshold = settings.limiter_threshold
                left_channel = np.tanh(left_channel / threshold) * threshold
                right_channel = np.tanh(right_channel / threshold) * threshold
            
            # Write to output
            outdata[:, 0] = left_channel
            outdata[:, 1] = right_channel
            
            # Update position
            self.position += frames / sample_rate
            
            # Check if finished
            if self.position >= settings.duration:
                self.position = settings.duration
                raise sd.CallbackStop()
                
        except sd.CallbackStop:
            raise
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            outdata.fill(0)
    
    def _emit_position(self):
        """Emit current position for UI updates."""
        self.position_changed.emit(self.position)
        
        if self.position >= self.project.settings.duration:
            self.stop()
            self.playback_finished.emit()
    
    def _check_no_sources_enabled(self) -> bool:
        """Check if any sources are enabled and warn if not."""
        enabled = sum(1 for s in self.project.sources if s.enabled)
        if enabled == 0:
            logger.warning("No audio sources enabled - playback will be silent")
            return True
        return False
    
    def play(self):
        """Start or resume playback."""
        if self.playing:
            return
        
        # Warn if no sources enabled
        if self._check_no_sources_enabled():
            QMessageBox.warning(None, "No Sources Enabled", 
                              "No audio sources are enabled. Playback will be silent.")
        
        # Create thread-safe snapshots before starting
        with self._playback_lock:
            self._playback_snapshots, self._playback_settings = self.project.create_playback_snapshot()
        
        settings = self.project.settings
        
        # Try with configured settings first, then fall back to defaults
        sample_rates_to_try = [settings.sample_rate]
        if settings.sample_rate != DEFAULT_SAMPLE_RATE:
            sample_rates_to_try.append(DEFAULT_SAMPLE_RATE)
        if settings.sample_rate != 48000 and DEFAULT_SAMPLE_RATE != 48000:
            sample_rates_to_try.append(48000)
        
        buffer_sizes_to_try = [settings.buffer_size]
        if settings.buffer_size != DEFAULT_BUFFER_SIZE:
            buffer_sizes_to_try.append(DEFAULT_BUFFER_SIZE)
        if settings.buffer_size != 2048 and DEFAULT_BUFFER_SIZE != 2048:
            buffer_sizes_to_try.append(2048)
        
        last_error = None
        
        for sample_rate in sample_rates_to_try:
            for buffer_size in buffer_sizes_to_try:
                try:
                    self.stream = sd.OutputStream(
                        samplerate=sample_rate,
                        blocksize=buffer_size,
                        channels=2,
                        callback=self._audio_callback,
                        finished_callback=lambda: None
                    )
                    self.stream.start()
                    self.playing = True
                    self.timer.start(50)  # Update UI every 50ms
                    
                    # Log if we fell back to different settings
                    if sample_rate != settings.sample_rate or buffer_size != settings.buffer_size:
                        logger.info(f"Audio: Using fallback settings - {sample_rate}Hz, buffer {buffer_size}")
                    
                    logger.info("Audio playback started")
                    return  # Success
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to start audio with {sample_rate}Hz, buffer {buffer_size}: {e}")
                    if self.stream:
                        try:
                            self.stream.close()
                        except:
                            pass
                        self.stream = None
                    continue
        
        # All attempts failed
        error_msg = f"Failed to start audio playback.\n\nError: {str(last_error)}\n\n"
        error_msg += "Suggestions:\n"
        error_msg += "• Check that an audio output device is connected\n"
        error_msg += "• Try a different sample rate in Global Settings\n"
        error_msg += "• Try increasing the buffer size for better compatibility"
        
        logger.error(f"Audio playback failed: {last_error}")
        QMessageBox.critical(None, "Audio Error", error_msg)
    
    def pause(self):
        """Pause playback."""
        if not self.playing:
            return
        
        self.timer.stop()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            self.stream = None
        self.playing = False
        logger.info("Audio playback paused")
    
    def stop(self):
        """Stop playback and reset position."""
        self.pause()
        self.position = 0.0
        
        # Reset all source phases
        for source in self.project.sources:
            source.reset_phase()
        
        # Clear playback snapshots
        with self._playback_lock:
            self._playback_snapshots = []
            self._playback_settings = None
        
        self.position_changed.emit(0.0)
        logger.info("Audio playback stopped")
    
    def seek(self, position: float):
        """Seek to a specific position."""
        was_playing = self.playing
        if was_playing:
            self.pause()
        
        self.position = max(0, min(position, self.project.settings.duration))
        
        # Attempt to compute approximate phases at seek position (Bug fix #6)
        # This reduces audible clicks for non-sine waveforms
        sample_rate = self.project.settings.sample_rate
        for source in self.project.sources:
            if not source.enabled:
                source.reset_phase()
                continue
            
            # Get frequency at seek position
            lf, rf, _, _ = source.get_parameters_at_time(self.position)
            
            # Estimate phase based on position and frequency
            # This won't be perfect but reduces discontinuity
            source._phase_left = (2 * np.pi * lf * self.position) % (2 * np.pi)
            source._phase_right = (2 * np.pi * rf * self.position) % (2 * np.pi)
        
        self.position_changed.emit(self.position)
        
        if was_playing:
            self.play()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        
# =============================================================================
# UI Components
# =============================================================================

class ColorButton(QPushButton):
    """Button that displays and allows selection of a color."""
    
    color_changed = Signal(str)
    
    def __init__(self, color: str = "#3498db"):
        super().__init__()
        self._color = color
        self.setFixedSize(30, 30)
        self._update_style()
        self.clicked.connect(self._pick_color)
    
    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                border: 2px solid #555;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: #888;
            }}
        """)
    
    def _pick_color(self):
        from PySide6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(QColor(self._color), self)
        if color.isValid():
            self._color = color.name()
            self._update_style()
            self.color_changed.emit(self._color)
    
    def color(self) -> str:
        return self._color
    
    def set_color(self, color: str):
        self._color = color
        self._update_style()

class EventEditor(QDialog):
    """Dialog for editing a parameter event."""
    
    def __init__(self, event: ParameterEvent, parent=None):
        super().__init__(parent)
        self.param_event = event
        self.setWindowTitle("Edit Event")
        self.setMinimumWidth(400)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Start time
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 3600)
        self.start_spin.setDecimals(2)
        self.start_spin.setSuffix(" s")
        self.start_spin.setValue(self.param_event.start_time)
        form.addRow("Start Time:", self.start_spin)
        
        # Duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 3600)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(self.param_event.duration)
        form.addRow("Duration:", self.duration_spin)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        form.addRow(line)
        
        # Left frequency
        self.left_freq_spin = QDoubleSpinBox()
        self.left_freq_spin.setRange(MIN_FREQUENCY, MAX_FREQUENCY)
        self.left_freq_spin.setDecimals(2)
        self.left_freq_spin.setSuffix(" Hz")
        self.left_freq_spin.setValue(self.param_event.left_frequency)
        form.addRow("Left Frequency:", self.left_freq_spin)
        
        # Right frequency
        self.right_freq_spin = QDoubleSpinBox()
        self.right_freq_spin.setRange(MIN_FREQUENCY, MAX_FREQUENCY)
        self.right_freq_spin.setDecimals(2)
        self.right_freq_spin.setSuffix(" Hz")
        self.right_freq_spin.setValue(self.param_event.right_frequency)
        form.addRow("Right Frequency:", self.right_freq_spin)
        
        # Binaural beat display
        self.beat_label = QLabel()
        self._update_beat_display()
        form.addRow("Binaural Beat:", self.beat_label)
        
        self.left_freq_spin.valueChanged.connect(self._update_beat_display)
        self.right_freq_spin.valueChanged.connect(self._update_beat_display)
        
        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        form.addRow(line2)
        
        # Left volume
        self.left_vol_spin = QDoubleSpinBox()
        self.left_vol_spin.setRange(0, 1)
        self.left_vol_spin.setDecimals(2)
        self.left_vol_spin.setSingleStep(0.05)
        self.left_vol_spin.setValue(self.param_event.left_volume)
        form.addRow("Left Volume:", self.left_vol_spin)
        
        # Right volume
        self.right_vol_spin = QDoubleSpinBox()
        self.right_vol_spin.setRange(0, 1)
        self.right_vol_spin.setDecimals(2)
        self.right_vol_spin.setSingleStep(0.05)
        self.right_vol_spin.setValue(self.param_event.right_volume)
        form.addRow("Right Volume:", self.right_vol_spin)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _update_beat_display(self):
        beat = abs(self.right_freq_spin.value() - self.left_freq_spin.value())
        self.beat_label.setText(f"{beat:.2f} Hz")
    
    def get_values(self) -> dict:
        return {
            'start_time': self.start_spin.value(),
            'duration': self.duration_spin.value(),
            'left_frequency': self.left_freq_spin.value(),
            'right_frequency': self.right_freq_spin.value(),
            'left_volume': self.left_vol_spin.value(),
            'right_volume': self.right_vol_spin.value(),
        }

class SourceEditor(QWidget):
    """Panel for editing an audio source."""
    
    source_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source: Optional[AudioSource] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Source properties group
        props_group = QGroupBox("Source Properties")
        props_layout = QFormLayout(props_group)
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._on_name_changed)
        props_layout.addRow("Name:", self.name_edit)
        
        # Enabled
        self.enabled_check = QCheckBox()
        self.enabled_check.toggled.connect(self._on_enabled_changed)
        props_layout.addRow("Enabled:", self.enabled_check)
        
        # Color
        self.color_btn = ColorButton()
        self.color_btn.color_changed.connect(self._on_color_changed)
        props_layout.addRow("Color:", self.color_btn)
        
        # Transition style
        self.transition_combo = QComboBox()
        for style in TransitionStyle:
            self.transition_combo.addItem(style.value, style)
        self.transition_combo.currentIndexChanged.connect(self._on_transition_changed)
        props_layout.addRow("Transition:", self.transition_combo)
        
        # Waveform
        self.waveform_combo = QComboBox()
        for wf in WaveformType:
            self.waveform_combo.addItem(wf.value, wf)
        self.waveform_combo.currentIndexChanged.connect(self._on_waveform_changed)
        props_layout.addRow("Waveform:", self.waveform_combo)
        
        layout.addWidget(props_group)
        
        # Events group
        events_group = QGroupBox("Parameter Events")
        events_layout = QVBoxLayout(events_group)
        
        # Events table
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(6)
        self.events_table.setHorizontalHeaderLabels([
            "Start", "Duration", "L Freq", "R Freq", "L Vol", "R Vol"
        ])
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.events_table.doubleClicked.connect(self._edit_selected_event)
        events_layout.addWidget(self.events_table)
        
        # Event buttons
        btn_layout = QHBoxLayout()
        
        self.add_event_btn = QPushButton("Add Event")
        self.add_event_btn.clicked.connect(self._add_event)
        btn_layout.addWidget(self.add_event_btn)
        
        self.edit_event_btn = QPushButton("Edit Event")
        self.edit_event_btn.clicked.connect(self._edit_selected_event)
        btn_layout.addWidget(self.edit_event_btn)
        
        self.delete_event_btn = QPushButton("Delete Event")
        self.delete_event_btn.clicked.connect(self._delete_selected_event)
        btn_layout.addWidget(self.delete_event_btn)
        
        events_layout.addLayout(btn_layout)
        layout.addWidget(events_group)
        
        # Presets group
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        self.alpha_btn = QPushButton("Alpha (8-12 Hz)")
        self.alpha_btn.clicked.connect(lambda: self._apply_preset("alpha"))
        presets_layout.addWidget(self.alpha_btn)
        
        self.theta_btn = QPushButton("Theta (4-8 Hz)")
        self.theta_btn.clicked.connect(lambda: self._apply_preset("theta"))
        presets_layout.addWidget(self.theta_btn)
        
        self.delta_btn = QPushButton("Delta (0.5-4 Hz)")
        self.delta_btn.clicked.connect(lambda: self._apply_preset("delta"))
        presets_layout.addWidget(self.delta_btn)
        
        self.beta_btn = QPushButton("Beta (12-30 Hz)")
        self.beta_btn.clicked.connect(lambda: self._apply_preset("beta"))
        presets_layout.addWidget(self.beta_btn)
        
        layout.addWidget(presets_group)
        
        self.setEnabled(False)
    
    def set_source(self, source: Optional[AudioSource]):
        self.source = source
        self.setEnabled(source is not None)
        
        if source:
            self.name_edit.setText(source.name)
            self.enabled_check.setChecked(source.enabled)
            self.color_btn.set_color(source.color)
            
            # Set transition combo
            for i in range(self.transition_combo.count()):
                if self.transition_combo.itemData(i) == source.transition_style:
                    self.transition_combo.setCurrentIndex(i)
                    break
            
            # Set waveform combo
            for i in range(self.waveform_combo.count()):
                if self.waveform_combo.itemData(i) == source.waveform:
                    self.waveform_combo.setCurrentIndex(i)
                    break
            
            self._refresh_events_table()
    
    def _refresh_events_table(self):
        if not self.source:
            return
        
        self.events_table.clearSelection()
        self.events_table.setRowCount(len(self.source.events))
        for i, event in enumerate(self.source.get_sorted_events()):
            self.events_table.setItem(i, 0, QTableWidgetItem(f"{event.start_time:.2f}s"))
            self.events_table.setItem(i, 1, QTableWidgetItem(f"{event.duration:.2f}s"))
            self.events_table.setItem(i, 2, QTableWidgetItem(f"{event.left_frequency:.1f}"))
            self.events_table.setItem(i, 3, QTableWidgetItem(f"{event.right_frequency:.1f}"))
            self.events_table.setItem(i, 4, QTableWidgetItem(f"{event.left_volume:.2f}"))
            self.events_table.setItem(i, 5, QTableWidgetItem(f"{event.right_volume:.2f}"))
    
    def _on_name_changed(self, text: str):
        if self.source:
            self.source.name = text
            self.source_changed.emit()
    
    def _on_enabled_changed(self, enabled: bool):
        if self.source:
            self.source.enabled = enabled
            self.source_changed.emit()
    
    def _on_color_changed(self, color: str):
        if self.source:
            self.source.color = color
            self.source_changed.emit()
    
    def _on_transition_changed(self, index: int):
        if self.source:
            self.source.transition_style = self.transition_combo.itemData(index)
            self.source_changed.emit()
    
    def _on_waveform_changed(self, index: int):
        if self.source:
            self.source.waveform = self.waveform_combo.itemData(index)
            self.source_changed.emit()
    
    def _add_event(self):
        if not self.source:
            return
        
        # Create event starting after the last one
        start_time = 0.0
        if self.source.events:
            last = max(self.source.events, key=lambda e: e.end_time())
            start_time = last.end_time()
        
        event = ParameterEvent(start_time=start_time)
        dialog = EventEditor(event, self)
        
        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            for key, value in values.items():
                setattr(event, key, value)
            self.source.events.append(event)
            self.source.mark_events_changed()
            self._refresh_events_table()
            self.source_changed.emit()
            
            # Check for overlaps and warn user (Bug fix #2)
            self._check_and_warn_overlaps()
    
    def _edit_selected_event(self):
        if not self.source:
            return
        
        row = self.events_table.currentRow()
        if row < 0:
            return
        
        sorted_events = self.source.get_sorted_events()
        if row >= len(sorted_events):
            return
        
        event = sorted_events[row]
        dialog = EventEditor(event, self)
        
        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            for key, value in values.items():
                setattr(event, key, value)
            self.source.mark_events_changed()
            self._refresh_events_table()
            self.source_changed.emit()
            
            # Check for overlaps and warn user (Bug fix #2)
            self._check_and_warn_overlaps()
    
    def _check_and_warn_overlaps(self):
        """Check for overlapping events and show a warning if found."""
        if not self.source:
            return
        
        overlaps = self.source.check_overlaps()
        if overlaps:
            overlap_info = []
            for e1, e2 in overlaps[:3]:  # Show up to 3 overlaps
                overlap_info.append(
                    f"  • Event at {e1.start_time:.1f}s overlaps with event at {e2.start_time:.1f}s"
                )
            
            msg = "Warning: Overlapping events detected:\n\n"
            msg += "\n".join(overlap_info)
            if len(overlaps) > 3:
                msg += f"\n  ... and {len(overlaps) - 3} more"
            msg += "\n\nOverlapping events will be blended together during playback."
            
            QMessageBox.warning(self, "Overlapping Events", msg)
    
    def _delete_selected_event(self):
        if not self.source:
            return
        
        row = self.events_table.currentRow()
        if row < 0:
            return
        
        sorted_events = self.source.get_sorted_events()
        if row >= len(sorted_events):
            return
        
        event = sorted_events[row]
        self.source.events.remove(event)
        self.source.mark_events_changed()
        self._refresh_events_table()
        self.source_changed.emit()
    
    def _apply_preset(self, preset: str):
        if not self.source or not self.source.events:
            return
        
        # Apply preset to all events
        presets = {
            "alpha": (200, 10),   # Base freq, beat
            "theta": (180, 6),
            "delta": (150, 2),
            "beta": (250, 20),
        }
        
        base, beat = presets.get(preset, (200, 10))
        
        for event in self.source.events:
            event.left_frequency = base
            event.right_frequency = base + beat
        
        self.source.mark_events_changed()
        self._refresh_events_table()
        self.source_changed.emit()


class GlobalSettingsPanel(QWidget):
    """Panel for editing global settings."""
    
    settings_changed = Signal()
    
    def __init__(self, settings: GlobalSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Duration group
        duration_group = QGroupBox("Duration")
        duration_layout = QFormLayout(duration_group)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(self.settings.duration)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        duration_layout.addRow("Total Duration:", self.duration_spin)
        
        layout.addWidget(duration_group)
        
        # Master volume group
        volume_group = QGroupBox("Master Volume")
        volume_layout = QFormLayout(volume_group)
        
        self.left_vol_slider = QSlider(Qt.Horizontal)
        self.left_vol_slider.setRange(0, 100)
        self.left_vol_slider.setValue(int(self.settings.master_volume_left * 100))
        self.left_vol_slider.valueChanged.connect(self._on_left_vol_changed)
        self.left_vol_label = QLabel(f"{self.settings.master_volume_left:.0%}")
        vol_left_layout = QHBoxLayout()
        vol_left_layout.addWidget(self.left_vol_slider)
        vol_left_layout.addWidget(self.left_vol_label)
        volume_layout.addRow("Left Channel:", vol_left_layout)
        
        self.right_vol_slider = QSlider(Qt.Horizontal)
        self.right_vol_slider.setRange(0, 100)
        self.right_vol_slider.setValue(int(self.settings.master_volume_right * 100))
        self.right_vol_slider.valueChanged.connect(self._on_right_vol_changed)
        self.right_vol_label = QLabel(f"{self.settings.master_volume_right:.0%}")
        vol_right_layout = QHBoxLayout()
        vol_right_layout.addWidget(self.right_vol_slider)
        vol_right_layout.addWidget(self.right_vol_label)
        volume_layout.addRow("Right Channel:", vol_right_layout)
        
        layout.addWidget(volume_group)
        
        # Audio settings group
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QFormLayout(audio_group)
        
        self.sample_rate_combo = QComboBox()
        for rate in [22050, 44100, 48000, 96000]:
            self.sample_rate_combo.addItem(f"{rate} Hz", rate)
        self.sample_rate_combo.setCurrentText(f"{self.settings.sample_rate} Hz")
        self.sample_rate_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        audio_layout.addRow("Sample Rate:", self.sample_rate_combo)
        
        self.buffer_combo = QComboBox()
        for size in [256, 512, 1024, 2048, 4096]:
            self.buffer_combo.addItem(str(size), size)
        self.buffer_combo.setCurrentText(str(self.settings.buffer_size))
        self.buffer_combo.currentIndexChanged.connect(self._on_buffer_changed)
        audio_layout.addRow("Buffer Size:", self.buffer_combo)
        
        layout.addWidget(audio_group)
        
        # Limiter group
        limiter_group = QGroupBox("Limiter")
        limiter_layout = QFormLayout(limiter_group)
        
        self.limiter_check = QCheckBox()
        self.limiter_check.setChecked(self.settings.limiter_enabled)
        self.limiter_check.toggled.connect(self._on_limiter_toggled)
        limiter_layout.addRow("Enabled:", self.limiter_check)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.settings.limiter_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        limiter_layout.addRow("Threshold:", self.threshold_spin)
        
        layout.addWidget(limiter_group)
        
        layout.addStretch()
    
    def _on_duration_changed(self, value: float):
        self.settings.duration = value
        self.settings_changed.emit()
    
    def _on_left_vol_changed(self, value: int):
        self.settings.master_volume_left = value / 100
        self.left_vol_label.setText(f"{value}%")
        self.settings_changed.emit()
    
    def _on_right_vol_changed(self, value: int):
        self.settings.master_volume_right = value / 100
        self.right_vol_label.setText(f"{value}%")
        self.settings_changed.emit()
    
    def _on_sample_rate_changed(self, index: int):
        self.settings.sample_rate = self.sample_rate_combo.itemData(index)
        self.settings_changed.emit()
    
    def _on_buffer_changed(self, index: int):
        self.settings.buffer_size = self.buffer_combo.itemData(index)
        self.settings_changed.emit()
    
    def _on_limiter_toggled(self, checked: bool):
        self.settings.limiter_enabled = checked
        self.settings_changed.emit()
    
    def _on_threshold_changed(self, value: float):
        self.settings.limiter_threshold = value
        self.settings_changed.emit()
    
    def update_from_settings(self, settings: GlobalSettings):
        self.settings = settings
        self.duration_spin.setValue(settings.duration)
        self.left_vol_slider.setValue(int(settings.master_volume_left * 100))
        self.right_vol_slider.setValue(int(settings.master_volume_right * 100))
        self.sample_rate_combo.setCurrentText(f"{settings.sample_rate} Hz")
        self.buffer_combo.setCurrentText(str(settings.buffer_size))
        self.limiter_check.setChecked(settings.limiter_enabled)
        self.threshold_spin.setValue(settings.limiter_threshold)


class TimelineVisualization(QWidget):
    """Interactive visualization of parameter curves over time."""
    
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        # Create graphics layout widget
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')
        
        # Frequency plot
        self.freq_plot = self.graphics_widget.addPlot(row=0, col=0, title="Frequency (Hz)")
        self.freq_plot.setLabel('left', 'Frequency', units='Hz')
        self.freq_plot.setLabel('bottom', 'Time', units='s')
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot.addLegend()
        
        # Volume plot
        self.vol_plot = self.graphics_widget.addPlot(row=1, col=0, title="Volume")
        self.vol_plot.setLabel('left', 'Volume')
        self.vol_plot.setLabel('bottom', 'Time', units='s')
        self.vol_plot.showGrid(x=True, y=True, alpha=0.3)
        self.vol_plot.addLegend()
        
        # Link x-axes
        self.vol_plot.setXLink(self.freq_plot)
        
        # Playhead lines
        self.freq_playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.vol_playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.freq_plot.addItem(self.freq_playhead)
        self.vol_plot.addItem(self.vol_playhead)
        
        layout.addWidget(self.graphics_widget)
        
        # Store curve references
        self.freq_curves: dict[str, tuple] = {}  # source_id -> (left_curve, right_curve)
        self.vol_curves: dict[str, tuple] = {}
    
    def _calculate_adaptive_points(self, duration: float) -> int:
        """Calculate number of visualization points based on duration."""
        # Use more points for shorter durations to capture details,
        # but cap at VIZ_MAX_POINTS to prevent performance issues
        ideal_points = int(duration * VIZ_POINTS_PER_SECOND)
        return max(VIZ_MIN_POINTS, min(VIZ_MAX_POINTS, ideal_points))
    
    def update_visualization(self):
        """Regenerate all curves from project data."""
        # Clear existing curves
        self.freq_plot.clear()
        self.vol_plot.clear()
        
        # Re-add playheads
        self.freq_plot.addItem(self.freq_playhead)
        self.vol_plot.addItem(self.vol_playhead)
        
        self.freq_curves.clear()
        self.vol_curves.clear()
        
        if not self.project.sources:
            return
        
        # Generate time points with adaptive resolution
        duration = self.project.settings.duration
        num_points = self._calculate_adaptive_points(duration)
        times = np.linspace(0, duration, num_points)
        
        for source in self.project.sources:
            if not source.enabled:
                continue
            
            color = QColor(source.color)
            color_left = color
            color_right = color.darker(120)
            
            # Get parameters at each time point
            left_freqs = []
            right_freqs = []
            left_vols = []
            right_vols = []
            
            for t in times:
                lf, rf, lv, rv = source.get_parameters_at_time(t)
                left_freqs.append(lf)
                right_freqs.append(rf)
                left_vols.append(lv)
                right_vols.append(rv)
            
            # Create frequency curves
            pen_left = pg.mkPen(color_left, width=2)
            pen_right = pg.mkPen(color_right, width=2, style=Qt.DashLine)
            
            left_freq_curve = self.freq_plot.plot(
                times, left_freqs, pen=pen_left,
                name=f"{source.name} (L)"
            )
            right_freq_curve = self.freq_plot.plot(
                times, right_freqs, pen=pen_right,
                name=f"{source.name} (R)"
            )
            self.freq_curves[source.id] = (left_freq_curve, right_freq_curve)
            
            # Create volume curves
            left_vol_curve = self.vol_plot.plot(
                times, left_vols, pen=pen_left,
                name=f"{source.name} (L)"
            )
            right_vol_curve = self.vol_plot.plot(
                times, right_vols, pen=pen_right,
                name=f"{source.name} (R)"
            )
            self.vol_curves[source.id] = (left_vol_curve, right_vol_curve)
        
        # Set axis ranges
        self.freq_plot.setXRange(0, duration)
        self.vol_plot.setXRange(0, duration)
        self.vol_plot.setYRange(0, 1.1)
        
    def set_playhead_position(self, position: float):
        """Update playhead position."""
        self.freq_playhead.setPos(position)
        self.vol_playhead.setPos(position)

class PlaybackControls(QWidget):
    """Playback control panel."""
    
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    seek_requested = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Play button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_clicked)
        layout.addWidget(self.play_btn)
        
        # Pause button
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_clicked)
        layout.addWidget(self.pause_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_clicked)
        layout.addWidget(self.stop_btn)
        
        layout.addSpacing(20)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderReleased.connect(self._on_seek)
        layout.addWidget(self.position_slider, 1)
        
        # Time display
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(100)
        layout.addWidget(self.time_label)
    
    def _on_seek(self):
        # Convert slider value to time
        ratio = self.position_slider.value() / 1000
        self.seek_requested.emit(ratio)
    
    def set_position(self, current: float, total: float):
        """Update position display."""
        if total > 0:
            ratio = current / total
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(int(ratio * 1000))
            self.position_slider.blockSignals(False)
        
        current_min = int(current) // 60
        current_sec = int(current) % 60
        total_min = int(total) // 60
        total_sec = int(total) % 60
        self.time_label.setText(f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}")
    
    def set_playing(self, playing: bool):
        """Update button states based on playing state."""
        self.play_btn.setEnabled(not playing)
        self.pause_btn.setEnabled(playing)


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.project = Project()
        self.undo_stack = UndoStack()
        self.audio_engine: Optional[AudioEngine] = None
        self.current_file: Optional[Path] = None
        
        self._setup_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._connect_signals()
        
        # Initialize audio engine
        self._init_audio_engine()
        
        # Add a default source
        self._add_default_source()
        
        self.setWindowTitle("Binaural Audio Generator")
        self.resize(1400, 900)
    
    def _setup_ui(self):
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Main splitter (horizontal)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Source list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        sources_label = QLabel("Audio Sources")
        sources_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(sources_label)
        
        self.source_list = QListWidget()
        self.source_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.source_list.customContextMenuRequested.connect(self._show_source_context_menu)
        self.source_list.currentRowChanged.connect(self._on_source_selected)
        left_layout.addWidget(self.source_list)
        
        # Source buttons
        source_btn_layout = QHBoxLayout()
        
        self.add_source_btn = QPushButton("Add Source")
        self.add_source_btn.clicked.connect(self._add_source)
        source_btn_layout.addWidget(self.add_source_btn)
        
        self.delete_source_btn = QPushButton("Delete")
        self.delete_source_btn.clicked.connect(self._delete_selected_source)
        source_btn_layout.addWidget(self.delete_source_btn)
        
        self.duplicate_source_btn = QPushButton("Duplicate")
        self.duplicate_source_btn.clicked.connect(self._duplicate_selected_source)
        source_btn_layout.addWidget(self.duplicate_source_btn)
        
        left_layout.addLayout(source_btn_layout)
        
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        self.main_splitter.addWidget(left_panel)
        
        # Center panel - Tabs for Editor and Settings
        center_panel = QTabWidget()
        
        # Source editor tab
        self.source_editor = SourceEditor()
        center_panel.addTab(self.source_editor, "Source Editor")
        
        # Global settings tab
        self.settings_panel = GlobalSettingsPanel(self.project.settings)
        center_panel.addTab(self.settings_panel, "Global Settings")
        
        self.main_splitter.addWidget(center_panel)
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        viz_label = QLabel("Timeline Visualization")
        viz_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(viz_label)
        
        self.timeline = TimelineVisualization(self.project)
        right_layout.addWidget(self.timeline)
        
        self.main_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        self.main_splitter.setSizes([200, 400, 600])
        
        main_layout.addWidget(self.main_splitter)
        
        # Playback controls
        self.playback_controls = PlaybackControls()
        main_layout.addWidget(self.playback_controls)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _setup_menus(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Project As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Audio...", self)
        export_action.triggered.connect(self._export_audio)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self._redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh Visualization", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_visualization)
        view_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self):
        # Playback shortcuts
        QShortcut(QKeySequence("Space"), self, self._toggle_playback)
        QShortcut(QKeySequence("Escape"), self, self._stop_playback)
    
    def _connect_signals(self):
        # Source editor signals
        self.source_editor.source_changed.connect(self._on_source_changed)
        
        # Settings panel signals
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        
        # Playback control signals
        self.playback_controls.play_clicked.connect(self._play)
        self.playback_controls.pause_clicked.connect(self._pause)
        self.playback_controls.stop_clicked.connect(self._stop_playback)
        self.playback_controls.seek_requested.connect(self._seek)
    
    def _init_audio_engine(self):
        if self.audio_engine:
            self.audio_engine.cleanup()
        
        self.audio_engine = AudioEngine(self.project)
        self.audio_engine.position_changed.connect(self._on_position_changed)
        self.audio_engine.playback_finished.connect(self._on_playback_finished)
    
    def _add_default_source(self):
        """Add a default source with example parameters."""
        source = AudioSource(
            name="Binaural Source 1",
            color="#3498db",
            events=[
                ParameterEvent(
                    start_time=0,
                    duration=10,
                    left_frequency=200,
                    right_frequency=210,
                    left_volume=0.7,
                    right_volume=0.7
                ),
                ParameterEvent(
                    start_time=10,
                    duration=20,
                    left_frequency=250,
                    right_frequency=256,
                    left_volume=0.8,
                    right_volume=0.8
                ),
                ParameterEvent(
                    start_time=30,
                    duration=20,
                    left_frequency=180,
                    right_frequency=184,
                    left_volume=0.6,
                    right_volume=0.6
                ),
            ]
        )
        self.project.sources.append(source)
        self._refresh_source_list()
        self._refresh_visualization()
    
    def _refresh_source_list(self):
        """Refresh the source list widget."""
        current_row = self.source_list.currentRow()
        self.source_list.blockSignals(True)
        self.source_list.clear()
        
        for source in self.project.sources:
            item = QListWidgetItem(source.name)
            item.setData(Qt.UserRole, source.id)
            
            # Set color indicator
            color = QColor(source.color)
            item.setForeground(color)
            
            # Indicate disabled state
            if not source.enabled:
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            
            self.source_list.addItem(item)
        
        self.source_list.blockSignals(False)
        
        # Restore selection
        if 0 <= current_row < self.source_list.count():
            self.source_list.setCurrentRow(current_row)
        elif self.source_list.count() > 0:
            self.source_list.setCurrentRow(0)
    def _refresh_visualization(self):
        """Refresh the timeline visualization."""
        self.timeline.update_visualization()
    
    def _get_selected_source(self) -> Optional[AudioSource]:
        """Get the currently selected source."""
        row = self.source_list.currentRow()
        if row < 0 or row >= len(self.project.sources):
            return None
        
        item = self.source_list.item(row)
        source_id = item.data(Qt.UserRole)
        
        for source in self.project.sources:
            if source.id == source_id:
                return source
        return None
    
    def _on_source_selected(self, row: int):
        """Handle source selection change."""
        source = self._get_selected_source()
        self.source_editor.set_source(source)
    
    def _on_source_changed(self):
        """Handle source modification."""
        self._save_undo_state()
        self._refresh_source_list()
        self._refresh_visualization()
    
    def _on_settings_changed(self):
        """Handle settings modification."""
        self._save_undo_state()
        self._refresh_visualization()
        
        # Update playback controls
        self.playback_controls.set_position(0, self.project.settings.duration)
    
    def _add_source(self):
        """Add a new audio source."""
        self._save_undo_state()
        
        # Generate unique color
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", 
                  "#1abc9c", "#e91e63", "#00bcd4", "#ff5722", "#607d8b"]
        color = colors[len(self.project.sources) % len(colors)]
        
        source = AudioSource(
            name=f"Source {len(self.project.sources) + 1}",
            color=color
        )
        self.project.sources.append(source)
        
        self._refresh_source_list()
        self.source_list.setCurrentRow(len(self.project.sources) - 1)
        self._refresh_visualization()
    
    def _delete_selected_source(self):
        """Delete the selected source."""
        source = self._get_selected_source()
        if not source:
            return
        
        reply = QMessageBox.question(
            self, "Delete Source",
            f"Delete source '{source.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._save_undo_state()
            self.project.sources.remove(source)
            self._refresh_source_list()
            self._refresh_visualization()
    
    def _duplicate_selected_source(self):
        """Duplicate the selected source."""
        source = self._get_selected_source()
        if not source:
            return
        
        self._save_undo_state()
        
        # Create a deep copy
        new_source = AudioSource.from_dict(source.to_dict())
        new_source.id = str(uuid.uuid4())
        new_source.name = f"{source.name} (Copy)"
        
        # Give new IDs to events
        for event in new_source.events:
            event.id = str(uuid.uuid4())
        
        self.project.sources.append(new_source)
        self._refresh_source_list()
        self.source_list.setCurrentRow(len(self.project.sources) - 1)
        self._refresh_visualization()
    
    def _show_source_context_menu(self, pos):
        """Show context menu for source list."""
        menu = QMenu(self)
        
        add_action = menu.addAction("Add Source")
        add_action.triggered.connect(self._add_source)
        
        source = self._get_selected_source()
        if source:
            menu.addSeparator()
            
            duplicate_action = menu.addAction("Duplicate")
            duplicate_action.triggered.connect(self._duplicate_selected_source)
            
            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(self._delete_selected_source)
            
            menu.addSeparator()
            
            toggle_action = menu.addAction(
                "Disable" if source.enabled else "Enable"
            )
            toggle_action.triggered.connect(lambda: self._toggle_source_enabled(source))
        
        menu.exec(self.source_list.mapToGlobal(pos))
    
    def _toggle_source_enabled(self, source: AudioSource):
        """Toggle source enabled state."""
        self._save_undo_state()
        source.enabled = not source.enabled
        self._refresh_source_list()
        self._refresh_visualization()
        self.source_editor.set_source(source)
    
    # Playback methods
    def _play(self):
        if self.audio_engine:
            self.audio_engine.play()
            self.playback_controls.set_playing(True)
            self.statusBar().showMessage("Playing...")
    
    def _pause(self):
        if self.audio_engine:
            self.audio_engine.pause()
            self.playback_controls.set_playing(False)
            self.statusBar().showMessage("Paused")
    
    def _stop_playback(self):
        if self.audio_engine:
            self.audio_engine.stop()
            self.playback_controls.set_playing(False)
            self.statusBar().showMessage("Stopped")
    
    def _toggle_playback(self):
        if self.audio_engine:
            if self.audio_engine.playing:
                self._pause()
            else:
                self._play()
    
    def _seek(self, ratio: float):
        if self.audio_engine:
            position = ratio * self.project.settings.duration
            self.audio_engine.seek(position)
    
    def _on_position_changed(self, position: float):
        self.playback_controls.set_position(position, self.project.settings.duration)
        self.timeline.set_playhead_position(position)
    
    def _on_playback_finished(self):
        self.playback_controls.set_playing(False)
        self.statusBar().showMessage("Playback finished")
    
    # Undo/Redo methods
    def _save_undo_state(self):
        self.undo_stack.push(self.project.to_dict())
        self._update_undo_actions()
    
    def _undo(self):
        state = self.undo_stack.undo(self.project.to_dict())
        if state:
            self.project = Project.from_dict(state)
            # Reset all source phases to prevent audio desync (Bug fix #8)
            for source in self.project.sources:
                source.reset_phase()
            self._init_audio_engine()
            self._refresh_all()
    
    def _redo(self):
        state = self.undo_stack.redo(self.project.to_dict())
        if state:
            self.project = Project.from_dict(state)
            # Reset all source phases to prevent audio desync (Bug fix #8)
            for source in self.project.sources:
                source.reset_phase()
            self._init_audio_engine()
            self._refresh_all()
    
    def _update_undo_actions(self):
        self.undo_action.setEnabled(self.undo_stack.can_undo())
        self.redo_action.setEnabled(self.undo_stack.can_redo())
    
    def _refresh_all(self):
        self._refresh_source_list()
        self._refresh_visualization()
        self.settings_panel.update_from_settings(self.project.settings)
        self._update_undo_actions()
        
        source = self._get_selected_source()
        self.source_editor.set_source(source)
    
    # File operations
    def _new_project(self):
        reply = QMessageBox.question(
            self, "New Project",
            "Create a new project? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.project = Project()
            self.undo_stack.clear()
            self.current_file = None
            self._init_audio_engine()
            self._add_default_source()
            self._refresh_all()
            self.setWindowTitle("Binaural Audio Generator - New Project")
    
    def _open_project(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "Binaural Project (*.binaural);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.project = Project.from_dict(data)
                self.undo_stack.clear()
                self.current_file = Path(filename)
                self._init_audio_engine()
                self._refresh_all()
                self.setWindowTitle(f"Binaural Audio Generator - {self.current_file.name}")
                self.statusBar().showMessage(f"Opened: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project: {str(e)}")
    
    def _save_project(self):
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self._save_project_as()
    
    def _save_project_as(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "Binaural Project (*.binaural);;JSON Files (*.json)"
        )
        
        if filename:
            if not filename.endswith(('.binaural', '.json')):
                filename += '.binaural'
            self._save_to_file(Path(filename))
    
    def _save_to_file(self, path: Path):
        try:
            with open(path, 'w') as f:
                json.dump(self.project.to_dict(), f, indent=2)
            
            self.current_file = path
            self.setWindowTitle(f"Binaural Audio Generator - {path.name}")
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
    
    def _export_audio(self):
        """Export the project to a WAV file with progress dialog."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Audio", "",
            "WAV Files (*.wav)"
        )
        
        if not filename:
            return
        
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # Check write permissions and path validity
        try:
            export_path = Path(filename)
            export_dir = export_path.parent
            if not export_dir.exists():
                QMessageBox.critical(self, "Export Error", 
                                   f"Directory does not exist: {export_dir}")
                return
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Invalid path: {e}")
            return
        
        # Check if any sources are enabled
        enabled_count = sum(1 for s in self.project.sources if s.enabled)
        if enabled_count == 0:
            reply = QMessageBox.question(self, "No Sources Enabled", 
                              "No audio sources are enabled. Export will be silent.\n\nContinue anyway?",
                              QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Create progress dialog
        progress = QProgressDialog("Preparing export...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Exporting Audio")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()
        
        try:
            import wave
            
            settings = self.project.settings
            duration = settings.duration
            sample_rate = settings.sample_rate
            num_samples = int(duration * sample_rate)
            
            logger.info(f"Starting export: {filename}, {duration}s, {sample_rate}Hz, {num_samples} samples")
            
            progress.setLabelText("Generating audio...")
            QApplication.processEvents()
            
            left_channel = np.zeros(num_samples)
            right_channel = np.zeros(num_samples)
            
            # Reset all phases
            for source in self.project.sources:
                source.reset_phase()
            
            # Count enabled sources for consistent normalization
            enabled_sources = sum(1 for s in self.project.sources if s.enabled)
            
            # Process in chunks for efficiency
            chunk_size = 8192
            total_sources = enabled_sources if enabled_sources > 0 else 1
            source_progress = 0
            
            for source in self.project.sources:
                if progress.wasCanceled():
                    logger.info("Export cancelled by user")
                    progress.close()
                    return
                
                if not source.enabled:
                    continue
                
                source.reset_phase()
                progress.setLabelText(f"Processing: {source.name}...")
                
                num_chunks = (num_samples + chunk_size - 1) // chunk_size
                
                for chunk_idx, chunk_start in enumerate(range(0, num_samples, chunk_size)):
                    if progress.wasCanceled():
                        logger.info("Export cancelled by user")
                        progress.close()
                        return
                    chunk_end = min(chunk_start + chunk_size, num_samples)
                    chunk_len = chunk_end - chunk_start
                    
                    # Update progress
                    chunk_progress = (source_progress + (chunk_idx / num_chunks)) / total_sources
                    progress.setValue(int(chunk_progress * 90))  # Reserve 10% for finalization
                    QApplication.processEvents()
                    
                    # Time array for this chunk
                    t_start = chunk_start / sample_rate
                    t_end = chunk_end / sample_rate
                    times = np.linspace(t_start, t_end, chunk_len, endpoint=False)
                    
                    # Get parameters for chunk
                    left_freqs = np.zeros(chunk_len)
                    right_freqs = np.zeros(chunk_len)
                    left_vols = np.zeros(chunk_len)
                    right_vols = np.zeros(chunk_len)
                    
                    for i, t in enumerate(times):
                        lf, rf, lv, rv = source.get_parameters_at_time(t)
                        left_freqs[i] = lf
                        right_freqs[i] = rf
                        left_vols[i] = lv
                        right_vols[i] = rv
                    
                    # Generate phase-continuous waveforms
                    dt = 1.0 / sample_rate
                    
                    # Accumulate phases
                    left_phase_increments = 2 * np.pi * left_freqs * dt
                    right_phase_increments = 2 * np.pi * right_freqs * dt
                    
                    left_phases = np.zeros(chunk_len)
                    right_phases = np.zeros(chunk_len)
                    
                    for i in range(chunk_len):
                        source._phase_left += left_phase_increments[i]
                        source._phase_right += right_phase_increments[i]
                        source._phase_left %= (2 * np.pi)
                        source._phase_right %= (2 * np.pi)
                        left_phases[i] = source._phase_left
                        right_phases[i] = source._phase_right
                    
                    # Generate waveform using correct waveform type
                    source_left = self._generate_waveform_array(left_phases, source.waveform) * left_vols
                    source_right = self._generate_waveform_array(right_phases, source.waveform) * right_vols
                    
                    left_channel[chunk_start:chunk_end] += source_left
                    right_channel[chunk_start:chunk_end] += source_right
                
                source_progress += 1
            
            # Finalization phase
            progress.setLabelText("Finalizing audio...")
            progress.setValue(92)
            QApplication.processEvents()
            
            # Normalize consistently with playback
            if enabled_sources > 1:
                norm = 1.0 / np.sqrt(enabled_sources)
                left_channel *= norm
                right_channel *= norm
            
            # Apply master volume
            left_channel *= settings.master_volume_left
            right_channel *= settings.master_volume_right
            
            # Apply limiter
            if settings.limiter_enabled:
                threshold = settings.limiter_threshold
                left_channel = np.tanh(left_channel / threshold) * threshold
                right_channel = np.tanh(right_channel / threshold) * threshold
            
            progress.setValue(95)
            progress.setLabelText("Converting to WAV format...")
            QApplication.processEvents()
            
            # Convert to 16-bit with safe handling of silent audio
            left_max = np.abs(left_channel).max() if len(left_channel) > 0 else 0
            right_max = np.abs(right_channel).max() if len(right_channel) > 0 else 0
            max_val = max(left_max, right_max)
            
            if max_val > 0:
                scale = 32767 / max_val * 0.95  # Leave 5% headroom
            else:
                scale = 32767
                logger.warning("Exporting silent audio")
            
            left_int = (left_channel * scale).astype(np.int16)
            right_int = (right_channel * scale).astype(np.int16)
            
            # Interleave channels
            stereo = np.column_stack((left_int, right_int)).flatten()
            
            progress.setValue(98)
            progress.setLabelText("Writing file...")
            QApplication.processEvents()
            
            # Write WAV file
            try:
                with wave.open(filename, 'w') as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(stereo.tobytes())
            except IOError as e:
                raise IOError(f"Failed to write file: {e}. Check disk space and permissions.")
            
            # Reset phases again
            for source in self.project.sources:
                source.reset_phase()
            
            progress.setValue(100)
            progress.close()
            
            logger.info(f"Export completed: {filename}")
            self.statusBar().showMessage(f"Exported: {filename}")
            QMessageBox.information(self, "Export Complete", 
                                  f"Audio exported successfully!\n\nFile: {filename}\n"
                                  f"Duration: {duration:.1f}s\nSample rate: {sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            progress.close()
            QMessageBox.critical(self, "Export Error", f"Failed to export audio:\n\n{str(e)}")
    
    def _generate_waveform_array(self, phase: np.ndarray, waveform: WaveformType) -> np.ndarray:
        """Generate waveform samples from phase values (for export)."""
        if waveform == WaveformType.SINE:
            return np.sin(phase)
        elif waveform == WaveformType.TRIANGLE:
            return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif waveform == WaveformType.SAWTOOTH:
            return 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        elif waveform == WaveformType.SQUARE:
            return np.sign(np.sin(phase))
        elif waveform == WaveformType.SOFT_SQUARE:
            return np.tanh(SOFT_SQUARE_SHARPNESS * np.sin(phase))
        return np.sin(phase)
    
    def _show_about(self):
        QMessageBox.about(
            self, "About Binaural Audio Generator",
            """<h2>Binaural Audio Generator</h2>
            <p>Version 1.0</p>
            <p>A comprehensive tool for creating customizable binaural audio experiences.</p>
            <h3>Features:</h3>
            <ul>
            <li>Multiple independent audio sources</li>
            <li>Time-based parameter sequences</li>
            <li>Smooth transitions with various easing curves</li>
            <li>Real-time visualization</li>
            <li>Low-latency audio playback</li>
            <li>Project save/load</li>
            <li>Audio export to WAV</li>
            </ul>
            <p>Built with Python, PySide6, NumPy, and PyQtGraph.</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.audio_engine:
            self.audio_engine.cleanup()
        event.accept()


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    # Set high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Apply dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()