from math import ceil
from nutils import mesh, function as fn


def backstep_ref(length, height, width, meshwidth):
    nel_length = int(ceil(length / meshwidth))
    nel_height = int(ceil(height / meshwidth))
    nel_width = int(ceil(width / meshwidth))
    nel_down = int(ceil(1 / meshwidth))

    domain, ref_geom = mesh.multipatch(
        patches=[[[0,1],[3,4]], [[3,4],[6,7]], [[2,3],[5,6]]],
        nelems={
            (0,1): nel_height, (3,4): nel_height, (6,7): nel_height,
            (2,5): nel_length, (3,6): nel_length, (4,7): nel_length,
            (0,3): nel_width, (1,4): nel_width,
            (2,3): nel_down, (5,6): nel_down,
        },
        patchverts=[
            [-width, 0], [-width, height],
            [0, -1], [0, 0], [0, height],
            [length, -1], [length, 0], [length, height]
        ],
    )

    return domain, ref_geom
