
#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 model;
	vec4 lightPos;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;


out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
	outColor = inColor;
	outUV = inUV;
    outNormal = mat3(ubo.model) * inNormal;
	
	// set projected position from the UV coords
    vec4 tempPos = ubo.projection * ubo.model * vec4(inPos.xyz, 1.0);
    vec4 projPos = vec4(inUV * 2.0 - 1.0, tempPos.z / tempPos.w, 1.0); 
    projPos.y = -projPos.y;
    projPos.z = (projPos.z + projPos.w) / 2.0f;
    gl_Position = projPos;
}