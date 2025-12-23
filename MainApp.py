import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, simpledialog
from tkinter import filedialog
import json
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import exp
import uuid
import copy

# Constants
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
DEFAULT_FREQ = 220.0
DEFAULT_VOL = 1.0
DEFAULT_DURATION = 600.0  # 10 minutes
INTERP_OPTIONS = [
    'linear',
    'parabolic',
    'parabolic_inverse',
    'exponential',
    'exponential_inverse',
    'polynomial',
    'polynomial_inverse'
]

class Segment:
    def __init__(self, start_time=0.0, duration=10.0, freq_l=DEFAULT_FREQ, freq_r=DEFAULT_FREQ, vol_l=DEFAULT_VOL, vol_r=DEFAULT_VOL):
        self.start_time = start_time
        self.duration = duration
        self.freq_l = freq_l
        self.freq_r = freq_r
        self.vol_l = vol_l
        self.vol_r = vol_r
        self.sync_id = str(uuid.uuid4())

class Voice:
    def __init__(self, name="Voice", interpolation_type=INTERP_OPTIONS[0]):
        self.name = name
        self.segments = []  # List of Segment objects
        self.phase_l = 0.0
        self.phase_r = 0.0
        self.interpolation_type = interpolation_type
        self.group_id = None
        self.unlocked_sync_ids = set()  # set of sync_ids that are unlocked for this voice

    def add_segment(self, segment):
        self.segments.append(segment)
        self.sort_segments()

    def sort_segments(self):
        self.segments.sort(key=lambda s: s.start_time)

    def reset_phases(self):
        self.phase_l = 0.0
        self.phase_r = 0.0

    def get_values_at(self, t):
        if not self.segments:
            return 0.0, 0.0, 0.0, 0.0
        self.sort_segments()
        # Find the interval
        for i in range(len(self.segments) - 1):
            seg = self.segments[i]
            next_seg = self.segments[i + 1]
            end = next_seg.start_time
            if seg.start_time <= t < end:
                dur = end - seg.start_time
                if dur <= 0:
                    return seg.freq_l, seg.freq_r, seg.vol_l, seg.vol_r
                frac = (t - seg.start_time) / dur
                
                typ = self.interpolation_type
                if typ == 'linear':
                    norm = frac
                elif typ == 'parabolic':
                    norm = frac ** 2  # Quadratic ease-in
                elif typ == 'parabolic_inverse':
                    norm = 1 - (1 - frac) ** 2  # Quadratic ease-out
                elif typ == 'exponential':
                    if frac == 0:
                        norm = 0.0
                    elif frac == 1:
                        norm = 1.0
                    else:
                        k = 5.0
                        norm = (exp(k * frac) - 1) / (exp(k) - 1)
                elif typ == 'exponential_inverse':
                    if frac == 0:
                        norm = 0.0
                    elif frac == 1:
                        norm = 1.0
                    else:
                        k = 5.0
                        norm = 1 - (exp(k * (1 - frac)) - 1) / (exp(k) - 1)
                elif typ == 'polynomial':
                    norm = 3 * frac**2 - 2 * frac**3  # Cubic ease-in-out
                elif typ == 'polynomial_inverse':
                    norm = 3 * frac**2 - 2 * frac**3  # Same as polynomial since symmetric
                else:
                    norm = frac
                
                freq_l = seg.freq_l + norm * (next_seg.freq_l - seg.freq_l)
                freq_r = seg.freq_r + norm * (next_seg.freq_r - seg.freq_r)
                vol_l = seg.vol_l + norm * (next_seg.vol_l - seg.vol_l)
                vol_r = seg.vol_r + norm * (next_seg.vol_r - seg.vol_r)
                return freq_l, freq_r, vol_l, vol_r
        # After last segment or in last
        last_seg = self.segments[-1]
        if t >= last_seg.start_time:
            return last_seg.freq_l, last_seg.freq_r, last_seg.vol_l, last_seg.vol_r
        # Before first
        return 0.0, 0.0, 0.0, 0.0

# Global variables
voices = []  # List of Voice objects
total_duration = DEFAULT_DURATION
global_vol_l = 1.0
global_vol_r = 1.0
current_time = 0.0
is_playing = False
is_paused = False
stream = None
p = None  # PyAudio instance
selected_voice_index = -1
check_vars = {}  # To be defined in main UI

def get_group_voices(group_id):
    if group_id is None:
        return []
    return [v for v in voices if v.group_id == group_id]

def audio_callback(in_data, frame_count, time_info, status):
    global current_time
    data = np.zeros((frame_count, 2), dtype=np.float32)
    dt = 1.0 / SAMPLE_RATE
    for i in range(frame_count):
        t = current_time + i * dt
        sum_l = 0.0
        sum_r = 0.0
        sum_vol_l = 0.0
        sum_vol_r = 0.0
        for voice in voices:
            freq_l, freq_r, vol_l, vol_r = voice.get_values_at(t)
            if freq_l > SAMPLE_RATE / 2 or freq_r > SAMPLE_RATE / 2:
                continue
            sample_l = np.sin(voice.phase_l) * vol_l
            sample_r = np.sin(voice.phase_r) * vol_r
            sum_l += sample_l
            sum_r += sample_r
            sum_vol_l += vol_l
            sum_vol_r += vol_r
            voice.phase_l += 2 * np.pi * freq_l * dt
            voice.phase_r += 2 * np.pi * freq_r * dt
        if sum_vol_l > 1.0:
            sum_l /= sum_vol_l
        if sum_vol_r > 1.0:
            sum_r /= sum_vol_r
        sum_l *= global_vol_l
        sum_r *= global_vol_r
        sum_l = np.clip(sum_l, -1.0, 1.0)
        sum_r = np.clip(sum_r, -1.0, 1.0)
        data[i, 0] = sum_l
        data[i, 1] = sum_r
    current_time += frame_count * dt
    if current_time >= total_duration:
        return data.tobytes(), pyaudio.paComplete
    return data.tobytes(), pyaudio.paContinue

def start_playback():
    global is_playing, is_paused, stream, p, current_time
    if is_playing:
        return
    if is_paused:
        stream.start_stream()
        is_playing = True
        is_paused = False
    else:
        current_time = 0.0
        for voice in voices:
            voice.reset_phases()
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=SAMPLE_RATE,
                        output=True,
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=audio_callback)
        stream.start_stream()
        is_playing = True
        is_paused = False

def pause_playback():
    global is_playing, is_paused
    if not is_playing:
        return
    stream.stop_stream()
    is_playing = False
    is_paused = True

def stop_playback():
    global is_playing, is_paused, stream, p, current_time
    if stream:
        stream.stop_stream()
        stream.close()
        if p:
            p.terminate()
        stream = None
        p = None
    is_playing = False
    is_paused = False
    current_time = 0.0
    for voice in voices:
        voice.reset_phases()

# UI Functions
def add_voice():
    name = simpledialog.askstring("Add Voice", "Enter voice name:")
    if name:
        voice = Voice(name)
        voices.append(voice)
        update_voices_list()

def delete_voice():
    global selected_voice_index
    if selected_voice_index >= 0:
        voice = voices[selected_voice_index]
        if voice.group_id:
            # Delete all in group
            group_voices = get_group_voices(voice.group_id)
            for gv in group_voices:
                voices.remove(gv)
        else:
            voices.remove(voice)
        selected_voice_index = -1
        update_voices_list()
        update_segments_tree()
        update_voice_settings()

def select_voice(event):
    global selected_voice_index
    selection = voices_list.curselection()
    if selection:
        selected_voice_index = selection[0]
        update_segments_tree()
        update_voice_settings()
        update_plot()

def update_voice_settings():
    if selected_voice_index >= 0:
        voice = voices[selected_voice_index]
        for typ, var in check_vars.items():
            var.set(False)  # Reset
        if voice.group_id:
            group_voices = get_group_voices(voice.group_id)
            types = [v.interpolation_type for v in group_voices]
            for typ in types:
                check_vars[typ].set(True)
        else:
            check_vars[voice.interpolation_type].set(True)
    else:
        for var in check_vars.values():
            var.set(False)

def set_voice_settings():
    if selected_voice_index < 0:
        return
    selected = [typ for typ, var in check_vars.items() if var.get()]
    if not selected:
        messagebox.showwarning("Warning", "Select at least one interpolation type.")
        return
    master = voices[selected_voice_index]
    old_group_id = master.group_id
    if old_group_id:
        # Remove old linked voices
        voices[:] = [v for v in voices if v.group_id != old_group_id or v is master]
    if len(selected) == 1:
        master.interpolation_type = selected[0]
        master.group_id = None
        master.unlocked_sync_ids.clear()
    else:
        group_id = str(uuid.uuid4())
        master.interpolation_type = selected[0]
        master.group_id = group_id
        master.unlocked_sync_ids.clear()
        # Create linked voices for remaining types
        for typ in selected[1:]:
            slave = Voice(name=f"{master.name} ({typ})", interpolation_type=typ)
            slave.group_id = group_id
            slave.unlocked_sync_ids = set()
            # Duplicate segments
            for seg in master.segments:
                s_copy = copy.deepcopy(seg)
                slave.add_segment(s_copy)
            voices.append(slave)
    update_voices_list()
    update_segments_tree()
    update_voice_settings()
    update_plot()

def add_segment():
    if selected_voice_index < 0:
        messagebox.showwarning("Warning", "Select a voice first.")
        return
    voice = voices[selected_voice_index]
    default_start = 0.0
    if voice.segments:
        last_seg = voice.segments[-1]
        default_start = last_seg.start_time + last_seg.duration
    popup = tk.Toplevel(root)
    popup.title("Add Segment")

    tk.Label(popup, text="Start Time:").grid(row=0, column=0)
    start_entry = tk.Entry(popup)
    start_entry.grid(row=0, column=1)
    start_entry.insert(0, str(default_start))

    tk.Label(popup, text="Duration:").grid(row=1, column=0)
    dur_entry = tk.Entry(popup)
    dur_entry.grid(row=1, column=1)
    dur_entry.insert(0, "10.0")

    tk.Label(popup, text="Freq Left:").grid(row=2, column=0)
    fl_entry = tk.Entry(popup)
    fl_entry.grid(row=2, column=1)
    fl_entry.insert(0, str(DEFAULT_FREQ))

    tk.Label(popup, text="Freq Right:").grid(row=3, column=0)
    fr_entry = tk.Entry(popup)
    fr_entry.grid(row=3, column=1)
    fr_entry.insert(0, str(DEFAULT_FREQ))

    tk.Label(popup, text="Vol Left:").grid(row=4, column=0)
    vl_entry = tk.Entry(popup)
    vl_entry.grid(row=4, column=1)
    vl_entry.insert(0, str(DEFAULT_VOL))

    tk.Label(popup, text="Vol Right:").grid(row=5, column=0)
    vr_entry = tk.Entry(popup)
    vr_entry.grid(row=5, column=1)
    vr_entry.insert(0, str(DEFAULT_VOL))

    fl_slider = ttk.Scale(popup, from_=0, to=2000, orient="horizontal", command=lambda v: fl_entry.delete(0, tk.END) or fl_entry.insert(0, f"{float(v):.2f}"))
    fl_slider.set(DEFAULT_FREQ)
    fl_slider.grid(row=2, column=2)

    fr_slider = ttk.Scale(popup, from_=0, to=2000, orient="horizontal", command=lambda v: fr_entry.delete(0, tk.END) or fr_entry.insert(0, f"{float(v):.2f}"))
    fr_slider.set(DEFAULT_FREQ)
    fr_slider.grid(row=3, column=2)

    vl_slider = ttk.Scale(popup, from_=0, to=1, orient="horizontal", command=lambda v: vl_entry.delete(0, tk.END) or vl_entry.insert(0, f"{float(v):.2f}"))
    vl_slider.set(DEFAULT_VOL)
    vl_slider.grid(row=4, column=2)

    vr_slider = ttk.Scale(popup, from_=0, to=1, orient="horizontal", command=lambda v: vr_entry.delete(0, tk.END) or vr_entry.insert(0, f"{float(v):.2f}"))
    vr_slider.set(DEFAULT_VOL)
    vr_slider.grid(row=5, column=2)

    def save_segment():
        try:
            start = float(start_entry.get())
            dur = float(dur_entry.get())
            fl = float(fl_entry.get())
            fr = float(fr_entry.get())
            vl = float(vl_entry.get())
            vr = float(vr_entry.get())
            if start < 0 or dur <= 0 or fl < 0 or fr < 0 or vl < 0 or vl > 1 or vr < 0 or vr > 1:
                raise ValueError
            if fl > SAMPLE_RATE / 2 or fr > SAMPLE_RATE / 2:
                messagebox.showwarning("Warning", "High frequencies may cause aliasing.")
            seg = Segment(start, dur, fl, fr, vl, vr)
            voice.add_segment(seg)
            if voice.group_id:
                group_voices = get_group_voices(voice.group_id)
                sync_id = seg.sync_id
                for other in group_voices:
                    if other is voice:
                        continue
                    o_seg = copy.deepcopy(seg)
                    o_seg.sync_id = sync_id
                    other.add_segment(o_seg)
            update_segments_tree()
            update_plot()
            popup.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input.")

    tk.Button(popup, text="Save", command=save_segment).grid(row=6, column=1)

def edit_segment():
    if selected_voice_index < 0:
        return
    selection = segments_tree.selection()
    if not selection:
        return
    item = segments_tree.item(selection[0])
    values = item['values']
    index = segments_tree.index(selection[0])
    voice = voices[selected_voice_index]
    seg = voice.segments[index]
    popup = tk.Toplevel(root)
    popup.title("Edit Segment")

    tk.Label(popup, text="Start Time:").grid(row=0, column=0)
    start_entry = tk.Entry(popup)
    start_entry.grid(row=0, column=1)
    start_entry.insert(0, values[0])

    tk.Label(popup, text="Duration:").grid(row=1, column=0)
    dur_entry = tk.Entry(popup)
    dur_entry.grid(row=1, column=1)
    dur_entry.insert(0, values[1])

    tk.Label(popup, text="Freq Left:").grid(row=2, column=0)
    fl_entry = tk.Entry(popup)
    fl_entry.grid(row=2, column=1)
    fl_entry.insert(0, values[2])

    tk.Label(popup, text="Freq Right:").grid(row=3, column=0)
    fr_entry = tk.Entry(popup)
    fr_entry.grid(row=3, column=1)
    fr_entry.insert(0, values[3])

    tk.Label(popup, text="Vol Left:").grid(row=4, column=0)
    vl_entry = tk.Entry(popup)
    vl_entry.grid(row=4, column=1)
    vl_entry.insert(0, values[4])

    tk.Label(popup, text="Vol Right:").grid(row=5, column=0)
    vr_entry = tk.Entry(popup)
    vr_entry.grid(row=5, column=1)
    vr_entry.insert(0, values[5])

    fl_slider = ttk.Scale(popup, from_=0, to=2000, orient="horizontal", command=lambda v: fl_entry.delete(0, tk.END) or fl_entry.insert(0, f"{float(v):.2f}"))
    fl_slider.set(float(values[2]))
    fl_slider.grid(row=2, column=2)

    fr_slider = ttk.Scale(popup, from_=0, to=2000, orient="horizontal", command=lambda v: fr_entry.delete(0, tk.END) or fr_entry.insert(0, f"{float(v):.2f}"))
    fr_slider.set(float(values[3]))
    fr_slider.grid(row=3, column=2)

    vl_slider = ttk.Scale(popup, from_=0, to=1, orient="horizontal", command=lambda v: vl_entry.delete(0, tk.END) or vl_entry.insert(0, f"{float(v):.2f}"))
    vl_slider.set(float(values[4]))
    vl_slider.grid(row=4, column=2)

    vr_slider = ttk.Scale(popup, from_=0, to=1, orient="horizontal", command=lambda v: vr_entry.delete(0, tk.END) or vr_entry.insert(0, f"{float(v):.2f}"))
    vr_slider.set(float(values[5]))
    vr_slider.grid(row=5, column=2)

    def save_edit():
        try:
            start = float(start_entry.get())
            dur = float(dur_entry.get())
            fl = float(fl_entry.get())
            fr = float(fr_entry.get())
            vl = float(vl_entry.get())
            vr = float(vr_entry.get())
            if start < 0 or dur <= 0 or fl < 0 or fr < 0 or vl < 0 or vl > 1 or vr < 0 or vr > 1:
                raise ValueError
            if fl > SAMPLE_RATE / 2 or fr > SAMPLE_RATE / 2:
                messagebox.showwarning("Warning", "High frequencies may cause aliasing.")
            seg.start_time = start
            seg.duration = dur
            seg.freq_l = fl
            seg.freq_r = fr
            seg.vol_l = vl
            seg.vol_r = vr
            voice.sort_segments()
            if voice.group_id and seg.sync_id not in voice.unlocked_sync_ids:
                group_voices = get_group_voices(voice.group_id)
                for other in group_voices:
                    if other is voice:
                        continue
                    for o_seg in other.segments:
                        if o_seg.sync_id == seg.sync_id:
                            o_seg.start_time = seg.start_time
                            o_seg.duration = seg.duration
                            o_seg.freq_l = seg.freq_l
                            o_seg.freq_r = seg.freq_r
                            o_seg.vol_l = seg.vol_l
                            o_seg.vol_r = seg.vol_r
                            other.sort_segments()
                            break
            update_segments_tree()
            update_plot()
            popup.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input.")

    tk.Button(popup, text="Save", command=save_edit).grid(row=6, column=1)

def unlock_segment():
    if selected_voice_index < 0:
        return
    selection = segments_tree.selection()
    if not selection:
        return
    index = segments_tree.index(selection[0])
    voice = voices[selected_voice_index]
    seg = voice.segments[index]
    if voice.group_id and seg.sync_id:
        voice.unlocked_sync_ids.add(seg.sync_id)
        # Make local copy if needed, but since already separate, just mark
        messagebox.showinfo("Info", "Segment unlocked for local edits.")

def lock_segment():
    if selected_voice_index < 0:
        return
    selection = segments_tree.selection()
    if not selection:
        return
    index = segments_tree.index(selection[0])
    voice = voices[selected_voice_index]
    seg = voice.segments[index]
    if voice.group_id and seg.sync_id in voice.unlocked_sync_ids:
        # Propagate to others
        group_voices = get_group_voices(voice.group_id)
        for other in group_voices:
            if other is voice:
                continue
            for o_seg in other.segments:
                if o_seg.sync_id == seg.sync_id:
                    o_seg.start_time = seg.start_time
                    o_seg.duration = seg.duration
                    o_seg.freq_l = seg.freq_l
                    o_seg.freq_r = seg.freq_r
                    o_seg.vol_l = seg.vol_l
                    o_seg.vol_r = seg.vol_r
                    other.sort_segments()
                    break
        voice.unlocked_sync_ids.remove(seg.sync_id)
        messagebox.showinfo("Info", "Segment locked and synced across group.")

def delete_segment():
    if selected_voice_index < 0:
        return
    selection = segments_tree.selection()
    if not selection:
        return
    index = segments_tree.index(selection[0])
    voice = voices[selected_voice_index]
    seg = voice.segments[index]
    sync_id = seg.sync_id
    if sync_id and voice.group_id:
        group_voices = get_group_voices(voice.group_id)
        for v in group_voices:
            for s in v.segments[:]:
                if s.sync_id == sync_id:
                    v.segments.remove(s)
    else:
        voice.segments.remove(seg)
    update_segments_tree()
    update_plot()

def update_voices_list():
    voices_list.delete(0, tk.END)
    for voice in voices:
        voices_list.insert(tk.END, voice.name)

def update_segments_tree():
    segments_tree.delete(*segments_tree.get_children())
    if selected_voice_index >= 0:
        voice = voices[selected_voice_index]
        for seg in voice.segments:
            segments_tree.insert("", tk.END, values=(seg.start_time, seg.duration, seg.freq_l, seg.freq_r, seg.vol_l, seg.vol_r))

def update_global_settings():
    global total_duration, global_vol_l, global_vol_r
    try:
        total_duration = float(duration_entry.get())
        global_vol_l = float(vol_l_entry.get())
        global_vol_r = float(vol_r_entry.get())
        if global_vol_l < 0 or global_vol_l > 1 or global_vol_r < 0 or global_vol_r > 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Invalid global settings.")
    update_plot()

def update_plot():
    ax[0].clear()
    ax[1].clear()
    ts = np.linspace(0, total_duration, 1000)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, voice in enumerate(voices):
        freq_l = np.array([voice.get_values_at(t)[0] for t in ts])
        freq_r = np.array([voice.get_values_at(t)[1] for t in ts])
        vol_l = np.array([voice.get_values_at(t)[2] for t in ts])
        vol_r = np.array([voice.get_values_at(t)[3] for t in ts])
        color = colors[idx % len(colors)]
        ax[0].plot(ts, freq_l, color + '-', label=f"{voice.name} Left Freq")
        ax[0].plot(ts, freq_r, color + '--', label=f"{voice.name} Right Freq")
        ax[1].plot(ts, vol_l, color + '-', label=f"{voice.name} Left Vol")
        ax[1].plot(ts, vol_r, color + '--', label=f"{voice.name} Right Vol")
    ax[0].set_title("Frequencies")
    ax[0].legend()
    ax[1].set_title("Volumes")
    ax[1].legend()
    canvas.draw()

def update_current_time_label():
    if is_playing:
        current_time_label.config(text=f"Current Time: {current_time:.2f} / {total_duration:.2f}")
        if current_time >= total_duration:
            stop_playback()
    root.after(100, update_current_time_label)

def load_config():
    global total_duration, global_vol_l, global_vol_r
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if filename:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            total_duration = data.get('total_duration', DEFAULT_DURATION)
            global_vol_l = data.get('global_vol_l', 1.0)
            global_vol_r = data.get('global_vol_r', 1.0)
            voices.clear()
            for v_data in data.get('voices', []):
                voice = Voice(name=v_data.get('name', 'Voice'), interpolation_type=v_data.get('interpolation_type', INTERP_OPTIONS[0]))
                voice.group_id = v_data.get('group_id', None)
                voice.unlocked_sync_ids = set(v_data.get('unlocked_sync_ids', []))
                for s_data in v_data.get('segments', []):
                    seg = Segment(
                        s_data.get('start_time', 0.0),
                        s_data.get('duration', 10.0),
                        s_data.get('freq_l', DEFAULT_FREQ),
                        s_data.get('freq_r', DEFAULT_FREQ),
                        s_data.get('vol_l', DEFAULT_VOL),
                        s_data.get('vol_r', DEFAULT_VOL)
                    )
                    seg.sync_id = s_data.get('sync_id', str(uuid.uuid4()))
                    voice.add_segment(seg)
                voices.append(voice)
            update_voices_list()
            update_segments_tree()
            update_voice_settings()
            update_plot()
            duration_entry.delete(0, tk.END)
            duration_entry.insert(0, str(total_duration))
            vol_l_entry.delete(0, tk.END)
            vol_l_entry.insert(0, str(global_vol_l))
            vol_r_entry.delete(0, tk.END)
            vol_r_entry.insert(0, str(global_vol_r))
            messagebox.showinfo("Success", "Configuration loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

def save_config():
    data = {
        'total_duration': total_duration,
        'global_vol_l': global_vol_l,
        'global_vol_r': global_vol_r,
        'voices': []
    }
    for voice in voices:
        v_data = {
            'name': voice.name,
            'interpolation_type': voice.interpolation_type,
            'group_id': voice.group_id,
            'unlocked_sync_ids': list(voice.unlocked_sync_ids),
            'segments': []
        }
        for seg in voice.segments:
            s_data = {
                'start_time': seg.start_time,
                'duration': seg.duration,
                'freq_l': seg.freq_l,
                'freq_r': seg.freq_r,
                'vol_l': seg.vol_l,
                'vol_r': seg.vol_r,
                'sync_id': seg.sync_id
            }
            v_data['segments'].append(s_data)
        data['voices'].append(v_data)
    filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if filename:
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

# Main UI
root = tk.Tk()
root.title("Binaural Audio Generator")

# Voices frame
voices_frame = ttk.Frame(root)
voices_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(voices_frame, text="Voices").pack()
voices_list = tk.Listbox(voices_frame)
voices_list.pack()
voices_list.bind("<<ListboxSelect>>", select_voice)

tk.Button(voices_frame, text="Add Voice", command=add_voice).pack()
tk.Button(voices_frame, text="Delete Voice", command=delete_voice).pack()

# Voice settings
tk.Label(voices_frame, text="Select Interpolation Types:").pack()
interp_frame = ttk.Frame(voices_frame)
interp_frame.pack()

check_vars = {}
for opt in INTERP_OPTIONS:
    var = tk.BooleanVar()
    chk = ttk.Checkbutton(interp_frame, text=opt.capitalize(), variable=var)
    chk.pack(anchor='w')
    check_vars[opt] = var

tk.Button(voices_frame, text="Set Settings", command=set_voice_settings).pack()

# Segments frame
segments_frame = ttk.Frame(root)
segments_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(segments_frame, text="Segments").pack()
columns = ("start", "dur", "fl", "fr", "vl", "vr")
segments_tree = ttk.Treeview(segments_frame, columns=columns, show="headings")
segments_tree.heading("start", text="Start")
segments_tree.heading("dur", text="Dur")
segments_tree.heading("fl", text="Freq L")
segments_tree.heading("fr", text="Freq R")
segments_tree.heading("vl", text="Vol L")
segments_tree.heading("vr", text="Vol R")
segments_tree.pack()

tk.Button(segments_frame, text="Add Segment", command=add_segment).pack()
tk.Button(segments_frame, text="Edit Segment", command=edit_segment).pack()
tk.Button(segments_frame, text="Delete Segment", command=delete_segment).pack()
tk.Button(segments_frame, text="Unlock Segment", command=unlock_segment).pack()
tk.Button(segments_frame, text="Lock Segment", command=lock_segment).pack()

# Global settings
global_frame = ttk.Frame(root)
global_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Label(global_frame, text="Total Duration:").pack()
duration_entry = tk.Entry(global_frame)
duration_entry.insert(0, str(DEFAULT_DURATION))
duration_entry.pack()

tk.Label(global_frame, text="Global Vol L:").pack()
vol_l_entry = tk.Entry(global_frame)
vol_l_entry.insert(0, "1.0")
vol_l_entry.pack()

tk.Label(global_frame, text="Global Vol R:").pack()
vol_r_entry = tk.Entry(global_frame)
vol_r_entry.insert(0, "1.0")
vol_r_entry.pack()

tk.Button(global_frame, text="Update Globals", command=update_global_settings).pack()

tk.Button(global_frame, text="Load Config", command=load_config).pack()
tk.Button(global_frame, text="Save Config", command=save_config).pack()

# Playback controls
play_frame = ttk.Frame(root)
play_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Button(play_frame, text="Play", command=start_playback).pack()
tk.Button(play_frame, text="Pause", command=pause_playback).pack()
tk.Button(play_frame, text="Stop", command=stop_playback).pack()

current_time_label = tk.Label(play_frame, text="Current Time: 0.00 / 600.00")
current_time_label.pack()

# Plot
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Initial update
update_voice_settings()

# Start updating time label
update_current_time_label()

root.mainloop()