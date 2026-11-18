#include "application.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace application {

namespace {

struct Vec2 {
	float u, v;
};

struct Vec3 {
	float x, y, z;
};

struct Vec4 {
	float x, y, z, w;
};

struct Vertex {
	Vec3 pos;
	Vec3 normal;
	Vec2 uv;
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

Mat4 mat4LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
	const Vec3 f = vec3Normalize(vec3Sub(target, eye));
	const Vec3 s = vec3Normalize(vec3Cross(f, up));
	const Vec3 u = vec3Cross(s, f);

	Mat4 r = mat4Identity();
	r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
	r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
	r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
	r.m[12] = -vec3Dot(s, eye);
	r.m[13] = -vec3Dot(u, eye);
	r.m[14] =  vec3Dot(f, eye);
	return r;
}

Mat4 mat4Perspective(float fovy_rad, float aspect, float z_near, float z_far) {
	Mat4 r{};
	const float f = 1.0f / std::tan(fovy_rad * 0.5f);
	r.m[0] = f / aspect;
	r.m[5] = -f;
	r.m[10] = z_far / (z_near - z_far);
	r.m[11] = -1.0f;
	r.m[14] = (z_near * z_far) / (z_near - z_far);
	return r;
}

Mat4 mat4NormalMatrix(const Mat4& model) {
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

// Vertex & Index buffers in VRAM (GPU-only)
VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vma_vertex_allocation = VK_NULL_HANDLE;

VkBuffer vk_index_buffer = VK_NULL_HANDLE;
VmaAllocation vma_index_allocation = VK_NULL_HANDLE;
uint32_t index_count = 0;

// Uniform buffer (Dynamic UBO)
VkBuffer vk_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation vma_uniform_allocation = VK_NULL_HANDLE;
void* uniform_mapped_data = nullptr;

// Texture resources in VRAM (GPU-only)
VkImage vk_texture_image = VK_NULL_HANDLE;
VmaAllocation vma_texture_allocation = VK_NULL_HANDLE;
VkImageView vk_texture_view = VK_NULL_HANDLE;
VkSampler vk_texture_sampler = VK_NULL_HANDLE;

// One-time upload command pool
VkCommandPool vk_upload_command_pool = VK_NULL_HANDLE;

// UI state
bool auto_rotate = true;
float model_pos[3] = { 0.0f, 0.0f, 0.0f };
float model_rot_deg[3] = { 20.0f, 35.0f, 0.0f };
float model_scale[3] = { 1.0f, 1.0f, 1.0f };

float camera_pos[3] = { 0.0f, 0.0f, 5.0f };
float camera_yaw_deg = -90.0f;
float camera_pitch_deg = 0.0f;
float camera_fov_deg = 60.0f;

float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.22f;

float dir_light_dir[3] = { -0.5f, -0.8f, -0.6f };
float dir_light_color[3] = { 1.0f, 0.98f, 0.92f };
float dir_light_intensity = 1.1f;

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

VkCommandBuffer beginSingleTimeCommands() {
	auto& context = graphics::internal::context;

	const VkCommandBufferAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_upload_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(context.device, &alloc_info, &cmd);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);
	return cmd;
}

void endSingleTimeCommands(VkCommandBuffer cmd) {
	auto& context = graphics::internal::context;

	vkEndCommandBuffer(cmd);

	const VkSubmitInfo submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	vkQueueSubmit(context.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeCommandBuffers(context.device, vk_upload_command_pool, 1, &cmd);
}

// Upload arbitrary host data into a GPU-only (VRAM) VkBuffer via a staging buffer
bool createDeviceBufferWithStaging(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkBuffer& out_buffer, VmaAllocation& out_allocation) {
	auto& context = graphics::internal::context;

	// 1. Create CPU-visible staging buffer
	const VkBufferCreateInfo staging_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo staging_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VkBuffer staging_buffer = VK_NULL_HANDLE;
	VmaAllocation staging_alloc = VK_NULL_HANDLE;
	VmaAllocationInfo staging_mapped{};
	if (vmaCreateBuffer(context.allocator, &staging_info, &staging_alloc_info,
	                    &staging_buffer, &staging_alloc, &staging_mapped) != VK_SUCCESS) {
		return false;
	}

	std::memcpy(staging_mapped.pMappedData, data, size);
	vmaFlushAllocation(context.allocator, staging_alloc, 0, size);

	// 2. Create GPU-only (VRAM) destination buffer
	const VkBufferCreateInfo gpu_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo gpu_alloc_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateBuffer(context.allocator, &gpu_info, &gpu_alloc_info,
	                    &out_buffer, &out_allocation, nullptr) != VK_SUCCESS) {
		vmaDestroyBuffer(context.allocator, staging_buffer, staging_alloc);
		return false;
	}

	// 3. Record and execute GPU copy command
	VkCommandBuffer cmd = beginSingleTimeCommands();
	const VkBufferCopy copy_region = {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size,
	};
	vkCmdCopyBuffer(cmd, staging_buffer, out_buffer, 1, &copy_region);
	endSingleTimeCommands(cmd);

	vmaDestroyBuffer(context.allocator, staging_buffer, staging_alloc);
	return true;
}

void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                           VkImageLayout old_layout, VkImageLayout new_layout) {
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	VkPipelineStageFlags src_stage = 0;
	VkPipelineStageFlags dst_stage = 0;

	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}

	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);
}

// Load image from filesystem via stb_image and upload to VRAM VkImage
bool loadTextureImage(const char* path) {
	auto& context = graphics::internal::context;

	int tex_width = 0;
	int tex_height = 0;
	int tex_channels = 0;
	stbi_uc* pixels = stbi_load(path, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
	if (!pixels) {
		std::string parent_path = std::string("../") + path;
		pixels = stbi_load(parent_path.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
	}
	if (!pixels) {
		std::cerr << "Failed to load texture image: " << path << '\n';
		return false;
	}

	const VkDeviceSize image_size = static_cast<VkDeviceSize>(tex_width) * tex_height * 4;

	// 1. Create staging buffer for pixels
	const VkBufferCreateInfo staging_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = image_size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo staging_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VkBuffer staging_buffer = VK_NULL_HANDLE;
	VmaAllocation staging_alloc = VK_NULL_HANDLE;
	VmaAllocationInfo staging_mapped{};
	if (vmaCreateBuffer(context.allocator, &staging_info, &staging_alloc_info,
	                    &staging_buffer, &staging_alloc, &staging_mapped) != VK_SUCCESS) {
		stbi_image_free(pixels);
		return false;
	}

	std::memcpy(staging_mapped.pMappedData, pixels, image_size);
	vmaFlushAllocation(context.allocator, staging_alloc, 0, image_size);
	stbi_image_free(pixels);

	// 2. Create GPU-only VkImage in VRAM
	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.extent = { static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height), 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo image_alloc_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image_info, &image_alloc_info,
	                   &vk_texture_image, &vma_texture_allocation, nullptr) != VK_SUCCESS) {
		vmaDestroyBuffer(context.allocator, staging_buffer, staging_alloc);
		return false;
	}

	// 3. Copy pixels from staging buffer to VkImage and transition layouts
	VkCommandBuffer cmd = beginSingleTimeCommands();
	transitionImageLayout(cmd, vk_texture_image,
	                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	const VkBufferImageCopy region = {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = { 0, 0, 0 },
		.imageExtent = { static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height), 1 },
	};
	vkCmdCopyBufferToImage(cmd, staging_buffer, vk_texture_image,
	                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	transitionImageLayout(cmd, vk_texture_image,
	                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	endSingleTimeCommands(cmd);

	vmaDestroyBuffer(context.allocator, staging_buffer, staging_alloc);

	// 4. Create VkImageView
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_texture_image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &vk_texture_view) != VK_SUCCESS) {
		return false;
	}

	// 5. Create VkSampler
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	if (vkCreateSampler(context.device, &sampler_info, nullptr, &vk_texture_sampler) != VK_SUCCESS) {
		return false;
	}

	return true;
}

// Build regular dodecahedron geometry with procedural normals and UV coordinates
void buildDodecahedronMesh(std::vector<Vertex>& out_vertices, std::vector<uint32_t>& out_indices) {
	const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
	const float inv_phi = 1.0f / phi;

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

	// Regular pentagon UV coordinates in [0.02, 0.98]
	const float two_pi = 2.0f * 3.1415926535f;
	Vec2 pentagon_uvs[5]{};
	for (int k = 0; k < 5; ++k) {
		const float angle = two_pi * static_cast<float>(k) / 5.0f;
		pentagon_uvs[k] = {
			0.5f + 0.48f * std::cos(angle),
			0.5f + 0.48f * std::sin(angle),
		};
	}

	out_vertices.clear();
	out_indices.clear();
	out_vertices.reserve(12 * 5);
	out_indices.reserve(12 * 9);

	for (int f = 0; f < 12; ++f) {
		const Vec3& p0 = v[faces[f][0]];
		const Vec3& p1 = v[faces[f][1]];
		const Vec3& p2 = v[faces[f][2]];

		Vec3 e1 = vec3Sub(p1, p0);
		Vec3 e2 = vec3Sub(p2, p0);
		Vec3 normal = vec3Normalize(vec3Cross(e1, e2));

		Vec3 center{ 0.0f, 0.0f, 0.0f };
		for (int k = 0; k < 5; ++k) {
			center = vec3Add(center, v[faces[f][k]]);
		}
		bool flip_winding = false;
		if (vec3Dot(normal, center) < 0.0f) {
			normal = { -normal.x, -normal.y, -normal.z };
			flip_winding = true;
		}

		const uint32_t base_vertex = static_cast<uint32_t>(out_vertices.size());
		for (int k = 0; k < 5; ++k) {
			out_vertices.push_back({ v[faces[f][k]], normal, pentagon_uvs[k] });
		}

		// Triangulate pentagon via fan using index buffer
		for (uint32_t t = 1; t <= 3; ++t) {
			if (!flip_winding) {
				out_indices.push_back(base_vertex + 0);
				out_indices.push_back(base_vertex + t);
				out_indices.push_back(base_vertex + t + 1);
			} else {
				out_indices.push_back(base_vertex + 0);
				out_indices.push_back(base_vertex + t + 1);
				out_indices.push_back(base_vertex + t);
			}
		}
	}
}

} // namespace

bool initialize() {
	auto& context = graphics::internal::context;

	// 1. Create temporary command pool for staging buffer transfers
	const VkCommandPoolCreateInfo upload_pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};
	if (vkCreateCommandPool(context.device, &upload_pool_info, nullptr, &vk_upload_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create upload command pool\n";
		return false;
	}

	// 2. Build dodecahedron mesh and upload Vertex + Index buffers to VRAM via staging
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	buildDodecahedronMesh(vertices, indices);
	index_count = static_cast<uint32_t>(indices.size());

	if (!createDeviceBufferWithStaging(vertices.data(), sizeof(Vertex) * vertices.size(),
	                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                                   vk_vertex_buffer, vma_vertex_allocation)) {
		std::cerr << "Failed to upload vertex buffer to VRAM\n";
		return false;
	}

	if (!createDeviceBufferWithStaging(indices.data(), sizeof(uint32_t) * indices.size(),
	                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                                   vk_index_buffer, vma_index_allocation)) {
		std::cerr << "Failed to upload index buffer to VRAM\n";
		return false;
	}

	// 3. Load albedo texture from filesystem into VRAM
	if (!loadTextureImage("textures/texture.png")) {
		std::cerr << "Failed to load textures/texture.png\n";
		return false;
	}

	// 4. Create Dynamic Uniform Buffer
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

	// 5. Create Descriptor Set Layout, Pool, and Set (Dynamic UBO + Combined Image Sampler)
	const VkDescriptorSetLayoutBinding bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = nullptr,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = nullptr,
		},
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 2,
		.pBindings = bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 2,
		.pPoolSizes = pool_sizes,
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

	const VkDescriptorImageInfo image_desc_info = {
		.sampler = vk_texture_sampler,
		.imageView = vk_texture_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};

	const VkWriteDescriptorSet descriptor_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &buffer_desc_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_descriptor_set,
			.dstBinding = 1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_desc_info,
		},
	};
	vkUpdateDescriptorSets(context.device, 2, descriptor_writes, 0, nullptr);

	// 6. Load SPIR-V shaders and build Graphics Pipeline
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
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, uv),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_desc,
		.vertexAttributeDescriptionCount = 3,
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
	if (vk_texture_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(context.device, vk_texture_sampler, nullptr);
	}
	if (vk_texture_view != VK_NULL_HANDLE) {
		vkDestroyImageView(context.device, vk_texture_view, nullptr);
	}
	if (vk_texture_image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, vk_texture_image, vma_texture_allocation);
	}
	if (vk_uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_uniform_buffer, vma_uniform_allocation);
	}
	if (vk_index_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_allocation);
	}
	if (vk_vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
	}
	if (vk_upload_command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(context.device, vk_upload_command_pool, nullptr);
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

	ImGui::Begin("Lab 3: Textured Dodecahedron");

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

	if (ImGui::CollapsingHeader("Material (Specular)", ImGuiTreeNodeFlags_DefaultOpen)) {
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

	const Mat4 m_scale = mat4Scale(model_scale[0], model_scale[1], model_scale[2]);
	const Mat4 m_rot_x = mat4RotateX(model_rot_deg[0] * deg_to_rad);
	const Mat4 m_rot_y = mat4RotateY(model_rot_deg[1] * deg_to_rad);
	const Mat4 m_rot_z = mat4RotateZ(model_rot_deg[2] * deg_to_rad);
	const Mat4 m_trans = mat4Translate(model_pos[0], model_pos[1], model_pos[2]);

	const Mat4 model = mat4Mul(m_trans, mat4Mul(m_rot_z, mat4Mul(m_rot_y, mat4Mul(m_rot_x, m_scale))));
	const Mat4 normal_matrix = mat4NormalMatrix(model);

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

	Vec3 norm_light_dir = vec3Normalize({ dir_light_dir[0], dir_light_dir[1], dir_light_dir[2] });

	SceneUBO ubo_data{};
	ubo_data.model = model;
	ubo_data.view_proj = view_proj;
	ubo_data.normal_matrix = normal_matrix;
	ubo_data.camera_pos = { eye.x, eye.y, eye.z, 1.0f };
	ubo_data.ambient_light = { ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity };
	ubo_data.dir_light_dir = { norm_light_dir.x, norm_light_dir.y, norm_light_dir.z, 0.0f };
	ubo_data.dir_light_color = { dir_light_color[0], dir_light_color[1], dir_light_color[2], dir_light_intensity };
	ubo_data.material_spec = { material_specular[0], material_specular[1], material_specular[2], material_shininess };

	std::memcpy(uniform_mapped_data, &ubo_data, sizeof(SceneUBO));
	vmaFlushAllocation(context.allocator, vma_uniform_allocation, 0, sizeof(SceneUBO));

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

	const uint32_t dynamic_offset = 0;
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_descriptor_set, 1, &dynamic_offset);

	const VkBuffer vertex_buffers[] = { vk_vertex_buffer };
	const VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application