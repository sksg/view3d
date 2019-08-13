#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model_matrix = mat4(1.0);
uniform mat4 view_matrix = mat4(1.0);

out vertex_data {
    vec3  normal;
    vec3  color;
} vertex_out;

void main() {
    mat4 vm = view_matrix * model_matrix;
    gl_Position = vm * vec4(position, 1.0);
    vertex_out.normal = normalize(mat3(vm) * normal);
    float falloff = dot(vertex_out.normal, vec3(0, 0, 1));
    if (falloff > 0)
        vertex_out.color = color * falloff;
    else
        vertex_out.color = color * 0.3;  // Inside/backside color
}