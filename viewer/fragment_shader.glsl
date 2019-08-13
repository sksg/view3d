#version 330
in fragment_data {
    vec2 texcoord;
    vec3 color;
} fragment_in;
out vec4 fragment_color;

void main() {
    float texdist = pow(fragment_in.texcoord.x, 2.0);
    texdist = texdist + pow(fragment_in.texcoord.y, 2.0);
    if (texdist > 1.0) discard;
    fragment_color = vec4(fragment_in.color, 1.0);
    // fragment_color = vec4(vec3(gl_FragCoord.z), 1.0);
}