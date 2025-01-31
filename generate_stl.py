import numpy as np
from scipy.special import sph_harm
from stl import mesh


def generate_stl(m, l, res_per_lobe_m, res_per_lobe_l, filename):

    _l = l - abs(m) + 1
    _m = 2 * abs(m)

    assert res_per_lobe_m % 2 == 0  # need to rotate by 45deg
    res_phi = res_per_lobe_l * _l
    res_theta = res_per_lobe_m * _m
    if m == 0:
        res_theta = res_per_lobe_m

    phi = np.linspace(0, np.pi, res_phi)
    theta = np.linspace(0, 2 * np.pi, res_theta)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # figure out the radius
    radius = np.abs(sph_harm(m, l, theta, phi).real)
    radius = radius / radius.max()
    radius += 0.1  # smooth a bit
    radius = radius / radius.max()
    radius *= 30 / 2  # max height/width of output
    _x = x * radius
    _y = y * radius
    _z = z * radius

    vertices = np.concatenate([_x[..., None], _y[..., None], _z[..., None]], axis=2)
    assert res_phi % _l == 0

    # fix vertices to make coloring easier
    for i in range(1, _l):
        vertices[:, i * res_per_lobe_l] *= 1.2
    if _m > 0:
        assert res_theta % _m == 0
        for i in range(_m):
            vertices[int((i + 0.5) * res_per_lobe_m) % res_theta, :] *= 1.2
            # vertices[int((i + 0.5) * res_per_lobe_m) % res_theta, :] += 1.2
    vertices = vertices.reshape(res_phi * res_theta, 3)

    def phi_theta_to_idx(_phi, _theta):
        return _theta * res_phi + _phi

    faces = []
    for _phi in range(res_phi - 1):
        _next_phi = _phi + 1
        for _theta in range(res_theta):
            _next_theta = (_theta + 1) % res_theta
            faces.append(
                [
                    phi_theta_to_idx(_phi, _theta),
                    phi_theta_to_idx(_phi, _next_theta),
                    phi_theta_to_idx(_next_phi, _theta),
                ]
            )
            faces.append(
                [
                    phi_theta_to_idx(_next_phi, _theta),
                    phi_theta_to_idx(_next_phi, _next_theta),
                    phi_theta_to_idx(_phi, _next_theta),
                ],
            )
    faces = np.vstack(faces)

    # Create the mesh
    sph_harm_stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            sph_harm_stl.vectors[i][j] = vertices[f[j], :]

    # Write the mesh to file "cube.stl"
    sph_harm_stl.save(filename)


for l in range(4):  # 9
    for m in range(l + 1):
        print(f"generating {l}.{m}")
        generate_stl(m, l, 160, 160, f"sphharm_m{m}_l{l}.stl")
