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

struct Vertex {
	Vec3 pos;
	Vec3 color;
};

struct Mat4 {
	float m[16];
};

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

// Orthographic projection for Vulkan clip space (Y down, Z in [0, 1])
Mat4 mat4Ortho(float left, float right, float bottom, float top, float z_near, float z_far) {
	Mat4 r = mat4Identity();
	r.m[0] = 2.0f / (right - left);
	r.m[5] = -2.0f / (top - bottom); // Flip Y for Vulkan
	r.m[10] = 1.0f / (z_near - z_far);
	r.m[12] = -(right + left) / (right - left);
	r.m[13] = -(top + bottom) / (top - bottom);
	r.m[14] = z_near / (z_near - z_far);
	return r;
}

// Vulkan resources
VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_pipeline = VK_NULL_HANDLE;

VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vma_vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

// UI state for interactive transformations & projection
bool use_orthographic = false;
bool auto_rotate = true;
float pos[3] = { 0.0f, 0.0f, -5.0f };
float rot_deg[3] = { 25.0f, 35.0f, 0.0f };
float scale_val = 1.0f;

std::vector<char> readShaderFile(const char* path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		// Try parent directory in case executable is launched from build-debug/
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

// Build regular dodecahedron geometry (12 pentagonal faces -> 36 triangles -> 108 vertices)
std::vector<Vertex> buildDodecahedronVertices() {
	const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f; // ~1.6180339887f
	const float inv_phi = 1.0f / phi;                  // ~0.6180339887f

	// 20 canonical vertices of a regular dodecahedron
	const Vec3 v[20] = {
		// (±1, ±1, ±1)
		{  1.0f,  1.0f,  1.0f }, // 0
		{  1.0f,  1.0f, -1.0f }, // 1
		{  1.0f, -1.0f,  1.0f }, // 2
		{  1.0f, -1.0f, -1.0f }, // 3
		{ -1.0f,  1.0f,  1.0f }, // 4
		{ -1.0f,  1.0f, -1.0f }, // 5
		{ -1.0f, -1.0f,  1.0f }, // 6
		{ -1.0f, -1.0f, -1.0f }, // 7

		// (0, ±1/phi, ±phi)
		{ 0.0f,  inv_phi,  phi }, // 8
		{ 0.0f,  inv_phi, -phi }, // 9
		{ 0.0f, -inv_phi,  phi }, // 10
		{ 0.0f, -inv_phi, -phi }, // 11

		// (±1/phi, ±phi, 0)
		{  inv_phi,  phi, 0.0f }, // 12
		{  inv_phi, -phi, 0.0f }, // 13
		{ -inv_phi,  phi, 0.0f }, // 14
		{ -inv_phi, -phi, 0.0f }, // 15

		// (±phi, 0, ±1/phi)
		{  phi, 0.0f,  inv_phi }, // 16
		{  phi, 0.0f, -inv_phi }, // 17
		{ -phi, 0.0f,  inv_phi }, // 18
		{ -phi, 0.0f, -inv_phi }, // 19
	};

	// 12 pentagonal faces (ordered around each face)
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

	// Distinct colors for the 12 pentagonal faces so 3D structure is clearly visible
	const Vec3 face_colors[12] = {
		{ 0.90f, 0.30f, 0.25f },
		{ 0.25f, 0.75f, 0.45f },
		{ 0.25f, 0.55f, 0.90f },
		{ 0.95f, 0.75f, 0.20f },
		{ 0.75f, 0.35f, 0.85f },
		{ 0.20f, 0.80f, 0.80f },
		{ 0.95f, 0.50f, 0.20f },
		{ 0.55f, 0.85f, 0.25f },
		{ 0.90f, 0.40f, 0.65f },
		{ 0.40f, 0.45f, 0.90f },
		{ 0.85f, 0.80f, 0.55f },
		{ 0.45f, 0.75f, 0.65f },
	};

	std::vector<Vertex> vertices;
	vertices.reserve(12 * 3 * 3);

	for (int f = 0; f < 12; ++f) {
		const Vec3 c = face_colors[f];
		const uint32_t i0 = faces[f][0];
		// Fan triangulation of pentagon: (0, 1, 2), (0, 2, 3), (0, 3, 4)
		for (int t = 1; t <= 3; ++t) {
			const uint32_t i1 = faces[f][t];
			const uint32_t i2 = faces[f][t + 1];
			vertices.push_back({ v[i0], c });
			vertices.push_back({ v[i1], c });
			vertices.push_back({ v[i2], c });
		}
	}

	return vertices;
}

} // namespace

bool initialize() {
	auto& context = graphics::internal::context;

	// 1. Create vertex buffer for the regular dodecahedron
	const std::vector<Vertex> vertices = buildDodecahedronVertices();
	vertex_count = static_cast<uint32_t>(vertices.size());
	const VkDeviceSize buffer_size = sizeof(Vertex) * vertices.size();

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = buffer_size,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo mapped_info{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
	                    &vk_vertex_buffer, &vma_vertex_allocation, &mapped_info) != VK_SUCCESS) {
		std::cerr << "Failed to create vertex buffer for dodecahedron\n";
		return false;
	}

	std::memcpy(mapped_info.pMappedData, vertices.data(), buffer_size);
	vmaFlushAllocation(context.allocator, vma_vertex_allocation, 0, buffer_size);

	// 2. Load compiled SPIR-V shaders
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

	// 3. Vertex input layout
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
			.offset = offsetof(Vertex, color),
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

	// Push constant for 4x4 MVP matrix
	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(Mat4),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
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
	if (vk_vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
	}
}

void update(double time) {
	static double last_time = time;
	const float dt = static_cast<float>(time - last_time);
	last_time = time;

	if (auto_rotate) {
		rot_deg[1] += dt * 45.0f;
		if (rot_deg[1] > 360.0f) {
			rot_deg[1] -= 360.0f;
		}
	}

	ImGui::Begin("Dodecahedron Controls (Lab 1)");
	ImGui::Checkbox("Orthographic Projection", &use_orthographic);
	ImGui::Checkbox("Auto Rotate", &auto_rotate);
	ImGui::SliderFloat3("Position", pos, -10.0f, 10.0f);
	ImGui::SliderFloat3("Rotation", rot_deg, 0.0f, 360.0f);
	ImGui::SliderFloat("Scale", &scale_val, 0.2f, 3.0f);
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	VkClearValue clear_values[2]{};
	clear_values[0].color = { { 0.10f, 0.12f, 0.15f, 1.0f } };
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

	const float deg_to_rad = 3.1415926535f / 180.0f;
	const Mat4 m_scale = mat4Scale(scale_val, scale_val, scale_val);
	const Mat4 m_rot_x = mat4RotateX(rot_deg[0] * deg_to_rad);
	const Mat4 m_rot_y = mat4RotateY(rot_deg[1] * deg_to_rad);
	const Mat4 m_rot_z = mat4RotateZ(rot_deg[2] * deg_to_rad);
	const Mat4 m_trans = mat4Translate(pos[0], pos[1], pos[2]);

	const Mat4 model = mat4Mul(m_trans, mat4Mul(m_rot_z, mat4Mul(m_rot_y, mat4Mul(m_rot_x, m_scale))));

	const float aspect = (context.swapchain_extent.height > 0)
		? static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height)
		: 1.0f;

	Mat4 proj{};
	if (use_orthographic) {
		const float ortho_size = 3.0f;
		proj = mat4Ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, 0.1f, 100.0f);
	} else {
		proj = mat4Perspective(60.0f * deg_to_rad, aspect, 0.1f, 100.0f);
	}

	const Mat4 mvp = mat4Mul(proj, model);

	vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &mvp);

	const VkBuffer vertex_buffers[] = { vk_vertex_buffer };
	const VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);

	vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application