#version 450

layout(location = 0) in vec2 v_position;
layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PushConstantData {
	mat3 camera_rotation;
	vec3 camera_position;
	float time;
	float aspect;
} pc;

//////////////////////////
// Consts
//////////////////////////

const int MAX_RAY_STEPS = 128;
const int MAX_SHADOW_STEPS = 64;

const int BASE = 1;

//////////////////////////
// Structs
//////////////////////////

struct Ray {
	vec3 origin;
	vec3 direction;
	int mask;
};

struct Hit {
	vec3 position;
	vec3 normal;
	float distance;
	int material;
};

struct Renderer {
	Hit hits[64];
	int num_hits;

	Ray rays[64];
	int num_rays;
};

//////////////////////////
// Functions
//////////////////////////

float map(in vec3 position, in int mask, out int material) {
	material = BASE;

	return length(position) - 0.5;
}

vec3 normal(in vec3 position, in int mask)
{
    vec2 e = vec2(1.0,-1.0)*0.5773;
    int material = 0;
    const float eps = 0.00025;
    return normalize(e.xyy * map(position + e.xyy * eps, mask, material) + 
					 e.yyx * map(position + e.yyx * eps, mask, material) + 
					 e.yxy * map(position + e.yxy * eps, mask, material) + 
					 e.xxx * map(position + e.xxx * eps, mask, material) );
}

bool intersect(in Ray ray, out Hit hit, out int material) {
	float dist = 0.0;

	for (int i = 0; i < MAX_RAY_STEPS; i++) {
		vec3 ray_position = ray.origin + ray.direction * dist;

		float d = map(ray_position, ray.mask, material);

		if (d < 0.001) {
			break;
		}

		if (dist > 500.0) {
			return false;
		}

		dist += d;
	}

	hit.position = ray.origin + ray.direction * dist;
	hit.distance = dist;
	hit.normal = normal(hit.position, ray.mask);
	hit.material = material;
	return true;
}

float shadow(in Ray ray, in float sharpness) {
	float dist = 0.0;
	float res = 1.0;

	for (int i = 0; i < MAX_SHADOW_STEPS; i++) {
		vec3 ray_position = ray.origin + ray.direction * dist;
		int material;

		float d = map(ray_position, ray.mask, material);

		if (res < 0.01) {
			break;
		}

		res = min(res, sharpness * d / dist);

		dist += d;
	}

	return clamp(res, 0.0, 1.0);
}

void add_pass(inout Renderer renderer, in Ray ray) {
	renderer.rays[renderer.num_rays] = ray;
	renderer.num_rays++;
}

vec3 render(in Ray ray) {
	vec3 color = vec3(0.0);

	Renderer renderer;
	renderer.num_hits = 0;
	renderer.num_rays = 0;
	add_pass(renderer, ray);
	
	for (int i = 0; i < renderer.num_rays; i++) {
		Hit hit;
		int material;

		if (intersect(renderer.rays[i], hit, material)) {
			renderer.hits[renderer.num_hits] = hit;
			renderer.num_hits += 1;
		}
	}

	if (renderer.num_hits > 0) {
		for (int i = 0; i >= 0; i--) {
			Hit hit = renderer.hits[i];
		
			switch (hit.material) {
				case BASE:
					color = vec3(1.0);
					break;
				
				default:

					break;
			}
		}
	} else {
		color = vec3(0.2, 0.2, 0.6);
	}

	color = pow(color, vec3(0.4545));

	return color;
}

void main() {
	vec3 ray_direction = vec3(v_position, 2.0);
	ray_direction.x *= pc.aspect;

	ray_direction = pc.camera_rotation * normalize(ray_direction);

	Ray ray;
	ray.origin = pc.camera_position;
	ray.direction = ray_direction;

    f_color = vec4(render(ray), 1.0);
}
