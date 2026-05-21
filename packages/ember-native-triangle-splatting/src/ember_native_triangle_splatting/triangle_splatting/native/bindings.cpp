// Triangle Splatting CUDA stage wrappers.
//
// The implementation calls the package-local staged copy of the official
// diff-triangle-rasterization kernels. This file keeps the Ember-facing native
// API descriptive while preserving the upstream kernels for parity checks.

#include <torch/extension.h>

#include "upstream/rasterize_points.h"

namespace ember_native_triangle_splatting::triangle_splatting_native {

torch::Tensor rendered_count_tensor(
    int rendered_instance_count,
    const torch::Tensor& reference_tensor) {
    auto rendered_count = torch::empty(
        {1},
        reference_tensor.options().dtype(torch::kInt32)
    );
    rendered_count.fill_(rendered_instance_count);
    return rendered_count;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_triangles_fwd_wrapper(
    const torch::Tensor& background_color,
    const torch::Tensor& triangle_vertices,
    const torch::Tensor& triangle_sigma,
    const torch::Tensor& vertices_per_triangle,
    const torch::Tensor& triangle_vertex_offsets,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& triangle_opacities,
    const torch::Tensor& view_matrix,
    const torch::Tensor& projection_matrix,
    const int num_triangles,
    const float tangent_fov_x,
    const float tangent_fov_y,
    const int image_height,
    const int image_width,
    const torch::Tensor& spherical_harmonics,
    const int sh_degree,
    const torch::Tensor& camera_position,
    const bool prefiltered,
    const bool debug) {
    // The upstream API treats scale and density-factor tensors as mutable
    // outputs. Allocate them here so the Python custom op remains functional.
    auto screen_space_scale = torch::zeros(
        {num_triangles},
        triangle_vertices.options().dtype(torch::kFloat32)
    );
    auto density_factor = torch::zeros(
        {num_triangles},
        triangle_vertices.options().dtype(torch::kFloat32)
    );

    auto result = RasterizetrianglesCUDA(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        precomputed_colors,
        triangle_opacities,
        screen_space_scale,
        density_factor,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        image_height,
        image_width,
        spherical_harmonics,
        sh_degree,
        camera_position,
        prefiltered,
        debug
    );

    return std::make_tuple(
        rendered_count_tensor(std::get<0>(result), triangle_vertices),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
        std::get<4>(result),
        std::get<5>(result),
        std::get<6>(result),
        std::get<7>(result),
        std::get<8>(result),
        std::get<9>(result)
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_triangles_bwd_wrapper(
    const torch::Tensor& background_color,
    const torch::Tensor& triangle_vertices,
    const torch::Tensor& triangle_sigma,
    const torch::Tensor& vertices_per_triangle,
    const torch::Tensor& triangle_vertex_offsets,
    const torch::Tensor& triangle_radii,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& view_matrix,
    const torch::Tensor& projection_matrix,
    const int num_triangles,
    const float tangent_fov_x,
    const float tangent_fov_y,
    const torch::Tensor& grad_rendered_image,
    const torch::Tensor& grad_auxiliary_image,
    const torch::Tensor& spherical_harmonics,
    const int sh_degree,
    const torch::Tensor& camera_position,
    const torch::Tensor& geometry_buffer,
    const torch::Tensor& rendered_count,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    const bool debug) {
    // Backward consumes the exact buffers created by forward.
    return RasterizetrianglesBackwardCUDA(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        triangle_radii,
        precomputed_colors,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        grad_rendered_image,
        grad_auxiliary_image,
        spherical_harmonics,
        sh_degree,
        camera_position,
        geometry_buffer,
        rendered_count.item<int>(),
        binning_buffer,
        image_buffer,
        debug
    );
}

torch::Tensor mark_visible_wrapper(
    torch::Tensor triangle_centers,
    torch::Tensor view_matrix,
    torch::Tensor projection_matrix) {
    // The upstream signature is non-const but only uses these tensors as inputs.
    return markVisible(triangle_centers, view_matrix, projection_matrix);
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_wrapper(
    torch::Tensor old_opacity,
    torch::Tensor old_scale,
    torch::Tensor relocation_counts,
    torch::Tensor binomial_coefficients,
    const int max_relocation_count) {
    return ComputeRelocationCUDA(
        old_opacity,
        old_scale,
        relocation_counts,
        binomial_coefficients,
        max_relocation_count
    );
}

}  // namespace ember_native_triangle_splatting::triangle_splatting_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    namespace native =
        ember_native_triangle_splatting::triangle_splatting_native;
    module.def("rasterize_triangles_fwd", &native::rasterize_triangles_fwd_wrapper);
    module.def("rasterize_triangles_bwd", &native::rasterize_triangles_bwd_wrapper);
    module.def("mark_visible", &native::mark_visible_wrapper);
    module.def("compute_relocation", &native::compute_relocation_wrapper);
}
