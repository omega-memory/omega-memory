"""Generate an animated terminal demo GIF for omega-memory.

Creates a simulated terminal session showing install, setup, and memory recall.
Each frame has an explicit duration -- no duplicate frames needed.
"""
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
WIDTH, HEIGHT = 800, 450

# Colors
BG = (13, 17, 23)          # #0d1117
GREEN = (63, 185, 80)      # #3fb950 -- prompt $
WHITE = (255, 255, 255)
GRAY = (110, 118, 129)     # #6e7681 -- comments
ACCENT = (88, 101, 242)    # #5865f2 -- response indicator
DIM_WHITE = (200, 205, 212)
TITLE_BAR_BG = (22, 27, 34)
BORDER = (48, 54, 61)
DOT_RED = (255, 95, 86)
DOT_YELLOW = (255, 189, 46)
DOT_GREEN = (39, 201, 63)
CURSOR_COLOR = (200, 210, 220)

# Fonts
FONT_MONO = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 16)
FONT_MONO_SM = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 14)

# Layout
PADDING_X = 24
TITLE_BAR_H = 36
LINE_HEIGHT = 22
CURSOR_W = 9
CURSOR_H = 18

# Timing (milliseconds)
CHAR_DELAY = 70       # ms per typed character
PAUSE_SHORT = 600     # ms
PAUSE_MEDIUM = 1200   # ms
PAUSE_LONG = 2000     # ms
HOLD_END = 4000       # ms
OUTPUT_LINE_DELAY = 150  # ms between response lines appearing


def make_base_frame():
    """Create the terminal window background (cached)."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Terminal window border
    draw.rounded_rectangle(
        [8, 8, WIDTH - 8, HEIGHT - 8],
        radius=12, fill=BG, outline=BORDER, width=1,
    )

    # Title bar
    draw.rounded_rectangle(
        [8, 8, WIDTH - 8, 8 + TITLE_BAR_H],
        radius=12, fill=TITLE_BAR_BG,
    )
    draw.rectangle(
        [8, 8 + TITLE_BAR_H - 12, WIDTH - 8, 8 + TITLE_BAR_H],
        fill=TITLE_BAR_BG,
    )
    draw.line(
        [(8, 8 + TITLE_BAR_H), (WIDTH - 8, 8 + TITLE_BAR_H)],
        fill=BORDER,
    )

    # Traffic light dots
    dot_y = 8 + TITLE_BAR_H // 2
    for i, color in enumerate([DOT_RED, DOT_YELLOW, DOT_GREEN]):
        cx = 28 + i * 22
        draw.ellipse([cx - 6, dot_y - 6, cx + 6, dot_y + 6], fill=color)

    # Title text
    title = "Terminal -- omega-memory demo"
    bbox = draw.textbbox((0, 0), title, font=FONT_MONO_SM)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, dot_y - 7), title, fill=GRAY, font=FONT_MONO_SM)

    return img


def content_y(line_idx):
    """Y position for a given line index."""
    return 8 + TITLE_BAR_H + 14 + line_idx * LINE_HEIGHT


def render_frame(base, lines, cursor_pos=None):
    """Render a frame with the given lines and optional cursor.

    Args:
        base: base terminal image
        lines: list of (color, text, prefix_color, prefix_text) tuples
               prefix_color/prefix_text can be None for no colored prefix
        cursor_pos: (line_idx, after_text) or None for no cursor
    """
    frame = base.copy()
    draw = ImageDraw.Draw(frame)

    for i, line_data in enumerate(lines):
        y = content_y(i)
        if y > HEIGHT - 20:
            break

        if len(line_data) == 4:
            color, text, prefix_color, prefix_text = line_data
        else:
            color, text = line_data
            prefix_color, prefix_text = None, None

        # Draw the full text in its color
        draw.text((PADDING_X, y), text, fill=color, font=FONT_MONO)

        # Overdraw the prefix in its special color
        if prefix_color and prefix_text:
            draw.text((PADDING_X, y), prefix_text, fill=prefix_color, font=FONT_MONO)

    # Draw cursor if specified
    if cursor_pos is not None:
        line_idx, after_text = cursor_pos
        y = content_y(line_idx)
        bbox = draw.textbbox((PADDING_X, y), after_text, font=FONT_MONO)
        cx = bbox[2] + 1
        draw.rectangle(
            [cx, y + 2, cx + CURSOR_W, y + CURSOR_H + 2],
            fill=CURSOR_COLOR,
        )

    return frame


def generate_gif():
    """Generate all frames with their durations and save the GIF."""
    base = make_base_frame()
    frames = []     # list of PIL Images
    durations = []  # list of ms durations

    # Current terminal state: list of line tuples
    # Each line: (color, full_text, prefix_color, prefix_text)
    lines = []

    def add_frame(duration_ms, cursor_pos=None):
        """Add a frame showing current lines state."""
        # GIF minimum duration is ~20ms; enforce minimum
        duration_ms = max(20, duration_ms)
        frames.append(render_frame(base, lines, cursor_pos))
        durations.append(duration_ms)

    def type_command(text, prefix="$ ", prefix_color=GREEN):
        """Simulate typing a command character by character."""
        line_idx = len(lines)
        for ci in range(len(text) + 1):
            partial = prefix + text[:ci]
            if line_idx < len(lines):
                lines[line_idx] = (WHITE, partial, prefix_color, prefix)
            else:
                lines.append((WHITE, partial, prefix_color, prefix))
            add_frame(CHAR_DELAY, cursor_pos=(line_idx, partial))

        # Final frame without cursor (brief hold)
        add_frame(100)

    def type_prompt(text, prefix="> ", prefix_color=ACCENT):
        """Simulate typing a user prompt."""
        type_command(text, prefix=prefix, prefix_color=prefix_color)

    def add_output(output_lines, delay_between=0):
        """Add output lines (appear instantly or with delay between them)."""
        if delay_between > 0:
            for color, text in output_lines:
                lines.append((color, text, None, None))
                add_frame(delay_between)
        else:
            for color, text in output_lines:
                lines.append((color, text, None, None))
            add_frame(100)

    def add_comment(text):
        """Add a gray comment line."""
        lines.append((GRAY, text, None, None))
        add_frame(100)

    def pause(ms):
        """Hold the current frame for a duration."""
        add_frame(ms)

    # =====================================================================
    # SCENE 1: Initial blank terminal
    # =====================================================================
    pause(800)

    # =====================================================================
    # SCENE 2: pip install omega-memory
    # =====================================================================
    type_command("pip install omega-memory")
    pause(PAUSE_SHORT)

    add_output([
        (DIM_WHITE, "Collecting omega-memory"),
        (DIM_WHITE, "  Downloading omega_memory-0.8.0-py3-none-any.whl"),
        (GREEN, "Successfully installed omega-memory-0.8.0"),
    ])
    pause(PAUSE_MEDIUM)

    # =====================================================================
    # SCENE 3: omega setup
    # =====================================================================
    type_command("omega setup")
    pause(PAUSE_SHORT)

    add_output([
        (GREEN,     "\u2713 Configured for Claude Code"),
        (GREEN,     "\u2713 Memory system ready"),
        (DIM_WHITE, ""),
        (DIM_WHITE, "Run `claude` to start with persistent memory."),
    ])
    pause(PAUSE_MEDIUM)

    # =====================================================================
    # SCENE 4: Time skip
    # =====================================================================
    pause(PAUSE_SHORT)
    add_comment("# 3 weeks later, new session...")
    pause(PAUSE_LONG)

    # =====================================================================
    # SCENE 5: User asks a question
    # =====================================================================
    type_prompt('"What auth approach did we decide on?"')
    pause(PAUSE_SHORT)

    # =====================================================================
    # SCENE 6: OMEGA response appears line by line
    # =====================================================================
    add_output([
        (ACCENT,    "\u2192 Decision (Feb 1):"),
        (WHITE,     '  "JWT tokens over session cookies'),
        (WHITE,     '   -- API needs stateless auth for'),
        (WHITE,     '   horizontal scaling."'),
        (DIM_WHITE, ""),
        (GRAY,      "  relevance: 0.94  |  accessed 7\u00d7"),
    ], delay_between=OUTPUT_LINE_DELAY)

    # =====================================================================
    # SCENE 7: Hold the final result
    # =====================================================================
    pause(HOLD_END)

    return frames, durations


def main():
    print("Generating frames...")
    frames, durations = generate_gif()
    print(f"Generated {len(frames)} frames")

    if not frames:
        print("ERROR: No frames generated!")
        return

    total_ms = sum(durations)
    print(f"Total duration: {total_ms / 1000:.1f}s")

    output_path = "/Users/singularityjason/Projects/omega-public/assets/demo.gif"
    print(f"Saving GIF to {output_path}...")

    # Convert frames to P mode (palette) for better GIF compatibility
    # Use the first frame's palette for all frames
    palette_frames = []
    for f in frames:
        palette_frames.append(f.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,  # Don't let PIL mess with our durations
    )

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved: {WIDTH}x{HEIGHT}, {len(frames)} frames, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
