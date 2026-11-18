#include "application.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

struct Vec3 {
	float x, y, z;
};

struct Vec4 {
	float x, y, z, w;
};

struct Vertex {
	Vec3 pos;
	Vec3 normal;
};

struct Mat4 {
	float m[16];
};

// Uniform buffer matching std140 layout in shaders
struct SceneUBO {
	Mat4 model;
	Mat4 view_proj;
	Mat4 normal_matrix;

	Vec4 camera_pos;      // xyz: camera position
	Vec4 ambient_light;   // rgb: ambient color, a: ambient intensity

	Vec4 dir_light_dir;   // xyz: light direction (from light to scene)
	Vec4 dir_light_color; // rgb: light color, a: light intensity

	Vec4 material_albedo; // rgb: diffuse albedo
	Vec4 material_spec;   // rgb: specular color, a: shininess
};

Vec3 vec3Add(const Vec3& a, const Vec3& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

Vec3 vec3Sub(const Vec3& a, const Vec3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

float vec3Dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3Cross(const Vec3& a, const Vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

Vec3 vec3Normalize(const Vec3& v) {
	const float len = std::sqrt(vec3Dot(v, v));
	if (len <= 1e-6f) {
		return { 0.0f, 1.0f, 0.0f };
	}
	return { v.x / len, v.y / len, v.z / len };
}

Mat4 mat4Identity() {
	Mat4 r{};
	r.m[0] = 1.0f;
	r.m[5] = 1.0f;
	r.m[10] = 1.0f;
	r.m[15] = 1.0f;
	return r;
}

// Column-major matrix multiplication: res = a * b
Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			r.m[col * 4 + row] =
				a.m[0 * 4 + row] * b.m[col * 4 + 0] +
				a.m[1 * 4 + row] * b.m[col * 4 + 1] +
				a.m[2 * 4 + row] * b.m[col * 4 + 2] +
				a.m[3 * 4 + row] * b.m[col * 4 + 3];
		}
	}
	return r;
}

Mat4 mat4Translate(float tx, float ty, float tz) {
	Mat4 r = mat4Identity();
	r.m[12] = tx;
	r.m[13] = ty;
	r.m[14] = tz;
	return r;
}

Mat4 mat4Scale(float sx, float sy, float sz) {
	Mat4 r{};
	r.m[0] = sx;
	r.m[5] = sy;
	r.m[10] = sz;
	r.m[15] = 1.0f;
	return r;
}

Mat4 mat4RotateX(float angle_rad) {
	Mat4 r = mat4Identity();
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	r.m[5] = c;
	r.m[6] = s;
	r.m[9] = -s;
	r.m[10] = c;
	return r;
}

Mat4 mat4RotateY(float angle_rad) {
	Mat4 r = mat4Identity();
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	r.m[0] = c;
	r.m[2] = -s;
	r.m[8] = s;
	r.m[10] = c;
	return r;
}

Mat4 mat4RotateZ(float angle_rad) {
	Mat4 r = mat4Identity();
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	r.m[0] = c;
	r.m[1] = s;
	r.m[4] = -s;
	r.m[5] = c;
	return r;
}

// View matrix from camera eye position and target look-at point
Mat4 mat4LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
	const Vec3 f = vec3Normalize(vec3Sub(target, eye));
	const Vec3 s = vec3Normalize(vec3Cross(f, up));
	const Vec3 u = vec3Cross(s, f);

	Mat4 r = mat4Identity();
	r.m[0] = s.x;
	r.m[4] = s.y;
	r.m[8] = s.z;

	r.m[1] = u.x;
	r.m[5] = u.y;
	r.m[9] = u.z;

	r.m[2] = -f.x;
	r.m[6] = -f.y;
	r.m[10] = -f.z;

	r.m[12] = -vec3Dot(s, eye);
	r.m[13] = -vec3Dot(u, eye);
	r.m[14] = vec3Dot(f, eye);
	return r;
}

// Perspective projection for Vulkan clip space (Y down, Z in [0, 1])
Mat4 mat4Perspective(float fovy_rad, float aspect, float z_near, float z_far) {
	Mat4 r{};
	const float f = 1.0f / std::tan(fovy_rad * 0.5f);
	r.m[0] = f / aspect;
	r.m[5] = -f; // Flip Y for Vulkan
	r.m[10] = z_far / (z_near - z_far);
	r.m[11] = -1.0f;
	r.m[14] = (z_near * z_far) / (z_near - z_far);
	return r;
}

// Compute inverse-transpose of upper-left 3x3 for normal transformation
Mat4 mat4NormalMatrix(const Mat4& model) {
	// Extract columns of upper 3x3
	const float a00 = model.m[0], a01 = model.m[4], a02 = model.m[8];
	const float a10 = model.m[1], a11 = model.m[5], a12 = model.m[9];
	const float a20 = model.m[2], a21 = model.m[6], a22 = model.m[10];

	const float c00 = a11 * a22 - a12 * a21;
	const float c01 = a12 * a20 - a10 * a22;
	const float c02 = a10 * a21 - a11 * a20;

	const float c10 = a02 * a21 - a01 * a22;
	const float c11 = a00 * a22 - a02 * a20;
	const float c12 = a01 * a20 - a00 * a21;

	const float c20 = a01 * a12 - a02 * a11;
	const float c21 = a02 * a10 - a00 * a12;
	const float c22 = a00 * a11 - a01 * a10;

	const float det = a00 * c00 + a01 * c01 + a02 * c02;
	if (std::abs(det) < 1e-6f) {
		return mat4Identity();
	}
	const float inv_det = 1.0f / det;

	// Inverse-transpose places cofactors / det directly in matching (row, col)
	Mat4 r = mat4Identity();
	r.m[0] = c00 * inv_det; r.m[4] = c01 * inv_det; r.m[8]  = c02 * inv_det;
	r.m[1] = c10 * inv_det; r.m[5] = c11 * inv_det; r.m[9]  = c12 * inv_det;
	r.m[2] = c20 * inv_det; r.m[6] = c21 * inv_det; r.m[10] = c22 * inv_det;
	return r;
}

// Vulkan resources
VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_pipeline = VK_NULL_HANDLE;

VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vma_vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

VkBuffer vk_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation vma_uniform_allocation = VK_NULL_HANDLE;
void* uniform_mapped_data = nullptr;

// Interactive UI state
// 1. Model transform
bool auto_rotate = true;
float model_pos[3] = { 0.0f, 0.0f, 0.0f };
float model_rot_deg[3] = { 20.0f, 35.0f, 0.0f };
float model_scale[3] = { 1.0f, 1.0f, 1.0f };

// 2. Camera parameters (Position, Rotation Yaw/Pitch/Roll, Field of View)
float camera_pos[3] = { 0.0f, 0.0f, 5.0f };
float camera_yaw_deg = -90.0f;  // -90 deg looks along -Z toward origin
float camera_pitch_deg = 0.0f;
float camera_fov_deg = 60.0f;

// 3. Ambient & Directional Light parameters
float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.18f;

float dir_light_dir[3] = { -0.5f, -0.8f, -0.6f };
float dir_light_color[3] = { 1.0f, 0.98f, 0.92f };
float dir_light_intensity = 1.1f;

// 4. Material parameters (Blinn-Phong: Albedo, Specular Color, Shininess)
float material_albedo[3] = { 0.22f, 0.58f, 0.88f };
float material_specular[3] = { 1.0f, 1.0f, 1.0f };
float material_shininess = 32.0f;

std::vector<char> readShaderFile(const char* path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::string parent_path = std::string("../") + path;
		file.open(parent_path, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Failed to open shader file: " << path << '\n';
			return {};
		}
	}

	const size_t file_size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(file_size);
	file.seekg(0);
	file.read(buffer.data(), static_cast<std::streamsize>(file_size));
	return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
	if (code.empty()) {
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	return module;
}

// Build regular dodecahedron with procedurally generated outward face normals
std::vector<Vertex> buildDodecahedronVertices() {
	const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
	const float inv_phi = 1.0f / phi;

	// 20 canonical vertices of a regular dodecahedron centered at origin
	const Vec3 v[20] = {
		{  1.0f,  1.0f,  1.0f }, // 0
		{  1.0f,  1.0f, -1.0f }, // 1
		{  1.0f, -1.0f,  1.0f }, // 2
		{  1.0f, -1.0f, -1.0f }, // 3
		{ -1.0f,  1.0f,  1.0f }, // 4
		{ -1.0f,  1.0f, -1.0f }, // 5
		{ -1.0f, -1.0f,  1.0f }, // 6
		{ -1.0f, -1.0f, -1.0f }, // 7

		{ 0.0f,  inv_phi,  phi }, // 8
		{ 0.0f,  inv_phi, -phi }, // 9
		{ 0.0f, -inv_phi,  phi }, // 10
		{ 0.0f, -inv_phi, -phi }, // 11

		{  inv_phi,  phi, 0.0f }, // 12
		{  inv_phi, -phi, 0.0f }, // 13
		{ -inv_phi,  phi, 0.0f }, // 14
		{ -inv_phi, -phi, 0.0f }, // 15

		{  phi, 0.0f,  inv_phi }, // 16
		{  phi, 0.0f, -inv_phi }, // 17
		{ -phi, 0.0f,  inv_phi }, // 18
		{ -phi, 0.0f, -inv_phi }, // 19
	};

	const uint32_t faces[12][5] = {
		{ 0,  8, 10,  2, 16 },
		{ 0, 16, 17,  1, 12 },
		{ 0, 12, 14,  4,  8 },
		{ 1,  9, 11,  3, 17 },
		{ 1, 12, 14,  5,  9 },
		{ 2, 10,  6, 15, 13 },
		{ 2, 13,  3, 17, 16 },
		{ 3, 11,  7, 15, 13 },
		{ 4, 14,  5, 19, 18 },
		{ 4, 18,  6, 10,  8 },
		{ 5,  9, 11,  7, 19 },
		{ 6, 18, 19,  7, 15 },
	};

	std::vector<Vertex> vertices;
	vertices.reserve(12 * 3 * 3);

	for (int f = 0; f < 12; ++f) {
		const Vec3& p0 = v[faces[f][0]];
		const Vec3& p1 = v[faces[f][1]];
		const Vec3& p2 = v[faces[f][2]];

		// Procedurally compute face normal via cross product of two edges
		Vec3 e1 = vec3Sub(p1, p0);
		Vec3 e2 = vec3Sub(p2, p0);
		Vec3 normal = vec3Normalize(vec3Cross(e1, e2));

		// Compute face center and ensure normal points outward from origin
		Vec3 center{ 0.0f, 0.0f, 0.0f };
		for (int k = 0; k < 5; ++k) {
			center = vec3Add(center, v[faces[f][k]]);
		}
		bool flip_winding = false;
		if (vec3Dot(normal, center) < 0.0f) {
			normal = { -normal.x, -normal.y, -normal.z };
			flip_winding = true;
		}

		// Triangulate pentagon into 3 triangles with consistent outward normal
		const uint32_t i0 = faces[f][0];
		for (int t = 1; t <= 3; ++t) {
			const uint32_t i1 = faces[f][t];
			const uint32_t i2 = faces[f][t + 1];
			if (!flip_winding) {
				vertices.push_back({ v[i0], normal });
				vertices.push_back({ v[i1], normal });
				vertices.push_back({ v[i2], normal });
			} else {
				vertices.push_back({ v[i0], normal });
				vertices.push_back({ v[i2], normal });
				vertices.push_back({ v[i1], normal });
			}
		}
	}

	return vertices;
}

} // namespace

bool initialize() {
	auto& context = graphics::internal::context;

	// 1. Create vertex buffer with positions and procedurally generated normals
	const std::vector<Vertex> vertices = buildDodecahedronVertices();
	vertex_count = static_cast<uint32_t>(vertices.size());
	const VkDeviceSize vb_size = sizeof(Vertex) * vertices.size();

	const VkBufferCreateInfo vb_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = vb_size,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo vb_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo vb_mapped{};
	if (vmaCreateBuffer(context.allocator, &vb_info, &vb_alloc_info,
	                    &vk_vertex_buffer, &vma_vertex_allocation, &vb_mapped) != VK_SUCCESS) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}

	std::memcpy(vb_mapped.pMappedData, vertices.data(), vb_size);
	vmaFlushAllocation(context.allocator, vma_vertex_allocation, 0, vb_size);

	// 2. Create Uniform Buffer for camera, lighting, and material parameters
	const VkBufferCreateInfo ubo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(SceneUBO),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo ubo_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo ubo_mapped{};
	if (vmaCreateBuffer(context.allocator, &ubo_info, &ubo_alloc_info,
	                    &vk_uniform_buffer, &vma_uniform_allocation, &ubo_mapped) != VK_SUCCESS) {
		std::cerr << "Failed to create uniform buffer\n";
		return false;
	}
	uniform_mapped_data = ubo_mapped.pMappedData;

	// 3. Create Descriptor Set Layout, Pool, and Set for the Uniform Buffer
	const VkDescriptorSetLayoutBinding ubo_layout_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &ubo_layout_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo ds_alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &ds_alloc_info, &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	const VkDescriptorBufferInfo buffer_desc_info = {
		.buffer = vk_uniform_buffer,
		.offset = 0,
		.range = sizeof(SceneUBO),
	};

	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_desc_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);

	// 4. Load SPIR-V shaders and build Graphics Pipeline
	const auto vert_code = readShaderFile("shaders/dodecahedron.vert.spv");
	const auto frag_code = readShaderFile("shaders/dodecahedron.frag.spv");

	VkShaderModule vert_module = createShaderModule(context.device, vert_code);
	VkShaderModule frag_module = createShaderModule(context.device, frag_code);
	if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
		std::cerr << "Failed to create shader modules\n";
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding_desc = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attr_descs[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, pos),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_desc,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attr_descs,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = nullptr,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	vkDestroyShaderModule(context.device, frag_module, nullptr);
	vkDestroyShaderModule(context.device, vert_module, nullptr);

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (vk_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	}
	if (vk_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	}
	if (vk_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	}
	if (vk_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);
	}
	if (vk_uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_uniform_buffer, vma_uniform_allocation);
	}
	if (vk_vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
	}
}

void update(double time) {
	static double last_time = time;
	const float dt = static_cast<float>(time - last_time);
	last_time = time;

	if (auto_rotate) {
		model_rot_deg[1] += dt * 35.0f;
		if (model_rot_deg[1] > 360.0f) {
			model_rot_deg[1] -= 360.0f;
		}
	}

	ImGui::Begin("Lab 2: Camera & Blinn-Phong Lighting");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("Camera Position", camera_pos, -15.0f, 15.0f);
		ImGui::SliderFloat("Camera Yaw (deg)", &camera_yaw_deg, -180.0f, 180.0f);
		ImGui::SliderFloat("Camera Pitch (deg)", &camera_pitch_deg, -89.0f, 89.0f);
		ImGui::SliderFloat("Field of View (deg)", &camera_fov_deg, 20.0f, 120.0f);
		if (ImGui::Button("Reset Camera")) {
			camera_pos[0] = 0.0f; camera_pos[1] = 0.0f; camera_pos[2] = 5.0f;
			camera_yaw_deg = -90.0f;
			camera_pitch_deg = 0.0f;
			camera_fov_deg = 60.0f;
		}
	}

	if (ImGui::CollapsingHeader("Directional & Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient Color", ambient_color);
		ImGui::SliderFloat("Ambient Intensity", &ambient_intensity, 0.0f, 1.0f);
		ImGui::Separator();
		ImGui::SliderFloat3("Light Direction", dir_light_dir, -1.0f, 1.0f);
		ImGui::ColorEdit3("Light Color", dir_light_color);
		ImGui::SliderFloat("Light Intensity", &dir_light_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Material (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo (Diffuse)", material_albedo);
		ImGui::ColorEdit3("Specular Color", material_specular);
		ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f);
	}

	if (ImGui::CollapsingHeader("Dodecahedron Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Auto Rotate", &auto_rotate);
		ImGui::SliderFloat3("Model Position", model_pos, -10.0f, 10.0f);
		ImGui::SliderFloat3("Model Rotation", model_rot_deg, 0.0f, 360.0f);
		ImGui::SliderFloat3("Model Scale", model_scale, 0.2f, 3.0f);
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const float deg_to_rad = 3.1415926535f / 180.0f;

	// 1. Compute Model and Normal matrices
	const Mat4 m_scale = mat4Scale(model_scale[0], model_scale[1], model_scale[2]);
	const Mat4 m_rot_x = mat4RotateX(model_rot_deg[0] * deg_to_rad);
	const Mat4 m_rot_y = mat4RotateY(model_rot_deg[1] * deg_to_rad);
	const Mat4 m_rot_z = mat4RotateZ(model_rot_deg[2] * deg_to_rad);
	const Mat4 m_trans = mat4Translate(model_pos[0], model_pos[1], model_pos[2]);

	const Mat4 model = mat4Mul(m_trans, mat4Mul(m_rot_z, mat4Mul(m_rot_y, mat4Mul(m_rot_x, m_scale))));
	const Mat4 normal_matrix = mat4NormalMatrix(model);

	// 2. Compute Camera View and Projection matrices
	const float yaw_rad = camera_yaw_deg * deg_to_rad;
	const float pitch_rad = camera_pitch_deg * deg_to_rad;

	const Vec3 cam_front = vec3Normalize({
		std::cos(yaw_rad) * std::cos(pitch_rad),
		std::sin(pitch_rad),
		std::sin(yaw_rad) * std::cos(pitch_rad),
	});
	const Vec3 eye = { camera_pos[0], camera_pos[1], camera_pos[2] };
	const Vec3 target = vec3Add(eye, cam_front);
	const Vec3 world_up = { 0.0f, 1.0f, 0.0f };

	const Mat4 view = mat4LookAt(eye, target, world_up);

	const float aspect = (context.swapchain_extent.height > 0)
		? static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height)
		: 1.0f;
	const Mat4 proj = mat4Perspective(camera_fov_deg * deg_to_rad, aspect, 0.1f, 100.0f);
	const Mat4 view_proj = mat4Mul(proj, view);

	// 3. Populate and flush SceneUBO
	Vec3 norm_light_dir = vec3Normalize({ dir_light_dir[0], dir_light_dir[1], dir_light_dir[2] });

	SceneUBO ubo_data{};
	ubo_data.model = model;
	ubo_data.view_proj = view_proj;
	ubo_data.normal_matrix = normal_matrix;
	ubo_data.camera_pos = { eye.x, eye.y, eye.z, 1.0f };
	ubo_data.ambient_light = { ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity };
	ubo_data.dir_light_dir = { norm_light_dir.x, norm_light_dir.y, norm_light_dir.z, 0.0f };
	ubo_data.dir_light_color = { dir_light_color[0], dir_light_color[1], dir_light_color[2], dir_light_intensity };
	ubo_data.material_albedo = { material_albedo[0], material_albedo[1], material_albedo[2], 1.0f };
	ubo_data.material_spec = { material_specular[0], material_specular[1], material_specular[2], material_shininess };

	std::memcpy(uniform_mapped_data, &ubo_data, sizeof(SceneUBO));
	vmaFlushAllocation(context.allocator, vma_uniform_allocation, 0, sizeof(SceneUBO));

	// 4. Record command buffer
	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	VkClearValue clear_values[2]{};
	clear_values[0].color = { { 0.08f, 0.09f, 0.12f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {
			.offset = { 0, 0 },
			.extent = context.swapchain_extent,
		},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);

	const VkBuffer vertex_buffers[] = { vk_vertex_buffer };
	const VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);

	vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application