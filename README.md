# Binaural Audio Generator

A comprehensive desktop application for creating customizable binaural audio experiences. Design stereo tones from multiple independent sources with parameters that evolve smoothly over time.

## Features

### Core Capabilities
- **Multiple Audio Sources**: Create unlimited independent stereo audio generators
- **Time-Based Parameter Sequences**: Define how frequency and volume change over time
- **Smooth Transitions**: 13 different easing curves for natural-sounding parameter changes
- **Multiple Waveforms**: Sine, Triangle, Sawtooth, Square, and Soft Square waves
- **Real-Time Playback**: Low-latency audio streaming with phase-continuous generation
- **Interactive Visualization**: See frequency and volume curves evolve in real-time

### Binaural Beat Presets
Quick presets for common brainwave entrainment frequencies:
- **Alpha (8-12 Hz)**: Relaxation, creativity
- **Theta (4-8 Hz)**: Meditation, deep relaxation
- **Delta (0.5-4 Hz)**: Deep sleep, healing
- **Beta (12-30 Hz)**: Focus, alertness

### Transition Styles
- Linear
- Ease In/Out (Quadratic)
- Ease In/Out (Cubic)
- Exponential In/Out
- Sine In/Out
- Step (Instant)

### Audio Processing
- Dynamic amplitude normalization for multiple sources
- Soft limiter to prevent clipping
- Configurable sample rate (22050 - 96000 Hz)
- Adjustable buffer size for latency optimization
- Independent left/right master volume controls

### Project Management
- Save/Load projects in JSON format
- Export audio to WAV files
- Full undo/redo support

## Installation

### Prerequisites
- Python 3.9 or higher
- Audio output device

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install PySide6 numpy sounddevice pyqtgraph
```

### Platform-Specific Notes

**Linux**: You may need to install PortAudio:
```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2

# Fedora
sudo dnf install portaudio
```

**macOS**: PortAudio should be installed automatically with sounddevice.

**Windows**: No additional dependencies required.

## Usage

### Running the Application

```bash
python binaural_app.py
```

### Quick Start Guide

1. **Create a Source**: Click "Add Source" in the left panel
2. **Edit Parameters**: Select the source and modify its properties:
   - Name and color for identification
   - Transition style for smooth changes
   - Waveform type
3. **Add Events**: Events define how parameters change over time
   - Set start time and duration
   - Configure left/right frequencies (difference creates the binaural beat)
   - Adjust left/right volumes
4. **Preview**: Click Play to hear your creation in real-time
5. **Export**: Save as a WAV file for use elsewhere

### Understanding Binaural Beats

Binaural beats occur when slightly different frequencies are played in each ear. The brain perceives a third tone - the "beat" - at the frequency difference.

Example: Left ear 200 Hz + Right ear 210 Hz = 10 Hz binaural beat

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Space | Play/Pause |
| Escape | Stop |
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| F5 | Refresh Visualization |

## Project File Format

Projects are saved as JSON files with the `.binaural` extension:

```json
{
  "name": "My Project",
  "sources": [...],
  "groups": [...],
  "settings": {
    "duration": 60.0,
    "master_volume_left": 1.0,
    "master_volume_right": 1.0,
    "sample_rate": 44100,
    "buffer_size": 1024,
    "limiter_enabled": true,
    "limiter_threshold": 0.95
  }
}
```

## Architecture

### Components

- **Data Models**: `Project`, `AudioSource`, `ParameterEvent`, `GlobalSettings`
- **Audio Engine**: Real-time synthesis with sounddevice
- **Visualization**: Interactive pyqtgraph timeline
- **UI**: PySide6 (Qt6) interface with dark theme

### Audio Generation

The engine generates audio in chunks using a callback-based approach:
1. For each sample, interpolate parameters using the configured easing function
2. Generate waveform with phase continuity across chunks
3. Mix all active sources with normalization
4. Apply soft limiting to prevent clipping

## Extending the Application

### Adding New Waveforms

Add to the `WaveformType` enum and implement in `generate_waveform()`:

```python
class WaveformType(Enum):
    # ... existing ...
    CUSTOM = "Custom"

def generate_waveform(self, phase, waveform):
    # ... existing ...
    elif waveform == WaveformType.CUSTOM:
        return your_custom_function(phase)
```

### Adding New Easing Functions

Add to `TransitionStyle` enum and `EASING_FUNCTIONS` dict:

```python
def ease_custom(t: float) -> float:
    return your_easing_calculation(t)

EASING_FUNCTIONS[TransitionStyle.CUSTOM] = ease_custom
```

## Troubleshooting

### No Sound Output
1. Check your system's audio output device
2. Verify volume sliders in the application
3. Ensure at least one source is enabled
4. Try increasing the buffer size in settings

### Audio Glitches/Crackling
1. Increase buffer size (Settings → Audio Settings)
2. Close other audio applications
3. Try a lower sample rate

### High CPU Usage
1. Reduce the number of active sources
2. Use simpler waveforms (sine instead of soft square)
3. Increase buffer size


## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with [PySide6](https://wiki.qt.io/Qt_for_Python)
- Audio streaming via [sounddevice](https://python-sounddevice.readthedocs.io/)
- Visualization powered by [pyqtgraph](https://www.pyqtgraph.org/)
