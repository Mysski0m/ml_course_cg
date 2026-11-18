#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

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
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);

    // Light direction pointing FROM fragment TO light source
    vec3 L = normalize(-ubo.dirLightDir.xyz);
    vec3 H = normalize(L + V);

    vec3 albedo = ubo.materialAlbedo.rgb;
    vec3 specColor = ubo.materialSpec.rgb;
    float shininess = max(ubo.materialSpec.a, 1.0);

    // 1. Ambient component
    vec3 ambient = (ubo.ambientLight.rgb * ubo.ambientLight.a) * albedo;

    // 2. Diffuse component (Lambert)
    float diff = max(dot(N, L), 0.0);
    vec3 lightRadiance = ubo.dirLightColor.rgb * ubo.dirLightColor.a;
    vec3 diffuse = diff * lightRadiance * albedo;

    // 3. Specular component (Blinn-Phong)
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(N, H), 0.0), shininess);
    }
    vec3 specular = spec * lightRadiance * specColor;

    vec3 result = ambient + diffuse + specular;
    outColor = vec4(result, 1.0);
}
