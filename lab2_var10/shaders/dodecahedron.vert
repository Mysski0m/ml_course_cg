#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 viewProj;
    mat4 normalMatrix;

    vec4 cameraPos;       // xyz: camera position
    vec4 ambientLight;    // rgb: ambient color, a: ambient intensity

    vec4 dirLightDir;     // xyz: light direction (from light to surface)
    vec4 dirLightColor;   // rgb: light color, a: light intensity

    vec4 materialAlbedo;  // rgb: diffuse albedo
    vec4 materialSpec;    // rgb: specular color, a: shininess
} ubo;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragNormal = mat3(ubo.normalMatrix) * inNormal;
    gl_Position = ubo.viewProj * worldPos;
}
