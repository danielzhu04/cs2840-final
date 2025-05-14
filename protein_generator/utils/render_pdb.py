import sys
from pymol import cmd, finish_launching

def render(pdb_path, png_path, width=400, height=400, ref_path=None):
    try:
        finish_launching(['pymol', '-cq'])
        if ref_path:
            cmd.load(ref_path, "ref")
            cmd.hide("everything", "ref")
        else:
            cmd.load("output/graphs/reference.pdb", "ref")
            cmd.hide("everything", "ref")
        cmd.load(pdb_path, "protein")
        cmd.dss("protein")
        cmd.hide("everything")
        cmd.show("cartoon", "protein")

        cmd.do("pair_fit protein and name CA and resi 1-5, ref and name CA and resi 1-5")
        cmd.cealign("protein", "ref")
        cmd.delete("ref")

        cmd.set_color("custom_purple", [177/255, 101/255, 181/255])  # RGB normalized to 0–1
        cmd.color("custom_purple", "protein")
        cmd.bg_color("white")
        cmd.set("ray_trace_mode", 1)
        cmd.set("ray_shadows", 0)
        cmd.set("antialias", 2)
        cmd.set("ray_opaque_background", 0)
        cmd.set("depth_cue", 0)
        cmd.set("cache_frames", 0) 
        cmd.set("field_of_view", 35)
        cmd.save("output/graphs/reference.pdb", "protein")
        cmd.png(png_path, width=width, height=height, dpi=300, ray=1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
    finally:
        cmd.quit()


if __name__ == "__main__":
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: render_pdb.py <in.pdb> <out.png> [<ref.pdb>]", file=sys.stderr)
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) == 4 else None
    render(inp, outp, ref_path=ref)