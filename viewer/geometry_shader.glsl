#version 330
layout (points) in;
layout (triangle_strip, max_vertices=4) out;

uniform mat4 perspective_matrix;
uniform float radius = 1.0;

in vertex_data {
    vec3 color;
    vec3 normal;
} vertex_in[];

out fragment_data {
    vec2 texcoord;
    vec3 color;
} vertex_out;

void main() {
    vertex_out.color = vertex_in[0].color;

    vec4 pos = gl_in[0].gl_Position;

    // Aligned with view port
    // vec3 u = vec3(1, 0, 0);
    // vec3 v = vec3(0, 1, 0);

    // Aligned with surface
    vec3 u, v;
    if (abs(vertex_in[0].normal.y) > abs(vertex_in[0].normal.x)) {
        v = cross(vertex_in[0].normal, vec3(1.0, 0.0, 0.0));
        u = cross(vertex_in[0].normal, v);
    } else {
        v = cross(vec3(0.0, 1.0, 0.0), vertex_in[0].normal);
        u = cross(vertex_in[0].normal, v);
    }

    u *= radius;
    v *= radius;

    vec4 a = pos + vec4(-u + v, 0.0);
    vec4 b = pos + vec4(-u - v, 0.0);
    vec4 c = pos + vec4(+u + v, 0.0);
    vec4 d = pos + vec4(+u - v, 0.0);

    gl_Position = perspective_matrix * a;
    vertex_out.texcoord = vec2(-1.0, 1.0);
    EmitVertex();

    gl_Position = perspective_matrix * b;
    vertex_out.texcoord = vec2(-1.0, -1.0);
    EmitVertex();

    gl_Position = perspective_matrix * c;
    vertex_out.texcoord = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = perspective_matrix * d;
    vertex_out.texcoord = vec2(1.0, -1.0);
    EmitVertex();
}