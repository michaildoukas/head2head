#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inViewVec;
layout (location = 2) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	outFragColor = vec4(inColor.rgb, 1.0);
}