
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
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;


out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
	//outNormal = inNormal;
	outColor = inColor;
	outUV = inUV;
    
    // camera transform
	vec4 pos = ubo.model * vec4(inPos.xyz, 1.0);
    
    // menpo perspective projection
    // ref https://github.com/menpo/menpo3d/blob/c7de9910f0382bf3a904c286db0255875d411081/menpo3d/camera.py#L82
    float f = ubo.projection[0].x;
    float c_x = ubo.projection[0].y;
    float c_y = ubo.projection[0].z;
    vec4 projPos;
    projPos.x = (f / pos.z) * pos.y + c_y;
    projPos.y = (f / pos.z) * pos.x + c_x;
    projPos.z = (pos.z - ubo.model[3].z);// / f;
    projPos.w = 1.f;
	
	// image space transforms
    vec2 resize = 	ubo.projection[1].xy;
    vec2 crop = 	ubo.projection[1].zw;
    mat4 projToScaleImg = mat4(
        resize.x,   0.f,        0.f, 0.f,
        0.f,        resize.y,   0.f, 0.f,
        0.f,        0.f,        1.f, 0.f,
        crop.x,     crop.y,     0.f, 1.f
    );
    projPos = projToScaleImg * projPos;
	
	// revert the axis swap
	float temp = projPos.x;
	projPos.x = projPos.y;
	projPos.y = temp;
	
	// above coords are in menpo image space
	// need to transform (x,y) to Vulkan space  
	float width = ubo.projection[2].x;
	float height = ubo.projection[2].y;
	float tx = -1.f;
	float ty = -1.f;
	mat4 menpoToVulkan;
	menpoToVulkan[0] = vec4(2.f/width, 	0.f, 			0.f, 0.f);
    menpoToVulkan[1] = vec4(0.f, 		2.f/height, 	0.f, 0.f);
	menpoToVulkan[2] = vec4(0.f, 		0.f, 			1.f, 0.f);
	menpoToVulkan[3] = vec4(tx, 		ty, 			0.f, 1.f);
	//projPos = menpoToVulkan * projPos;
	projPos.w = 500.f;
	
    // account for Vulkan NDC space
    // ref https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
	//projPos.y = -projPos.y;
	projPos.z = (projPos.z + projPos.w) / 2.0f;
	
    gl_Position = projPos;
	
    // compute lighting in original coords
	outNormal = inNormal;
	vec3 lPos = ubo.lightPos.xyz;
	outLightVec = lPos - inPos;
	outViewVec = -inPos;
}