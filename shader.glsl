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

const int MAX_RAY_STEPS = 256;

//////////////////////////
// Structs
//////////////////////////

struct Ray {
	vec3 origin;
	vec3 direction;
};

struct Hit {
	vec3 position;
	vec3 normal;
	float distance;
	int material;
};

//////////////////////////
// Functions
//////////////////////////

float map(in vec3 position) {
	return length(position) - 0.5;
}

bool intersect(in Ray ray, out Hit hit) {
	float dist = 0.0;

	for (int i = 0; i < MAX_RAY_STEPS; i++) {
		vec3 ray_position = ray.origin + ray.direction * dist;

		float d = map(ray_position);

		if (d < 0.01) {
			break;
		}

		if (dist > 500.0) {
			return false;
		}

		dist += d;
	}

	hit.position = ray.origin + ray.direction * dist;
	hit.distance = dist;
	return true;
}

float shadow(in Ray ray, in float sharpness) {
	float dist = 0.0;
	float res = 1.0;

	for (int i = 0; i < 64; i++) {
		vec3 ray_position = ray.origin + ray.direction * dist;

		float d = map(ray_position);

		if (res < 0.01) {
			break;
		}

		res = min(res, sharpness * d / dist);
	}

	return clamp(res, 0.0, 1.0);
}

vec3 render(in Ray ray) {
	vec3 color;
	
	vec3 light_direction = vec3(1.0, 0.0, 0.0);
	
	Hit hit;

	if (intersect(ray, hit)) {
		Ray shadow_ray;
		shadow_ray.origin = hit.position;
		shadow_ray.direction = light_direction;

		float shadow = shadow(ray, 4.0);

		color = vec3(1.0, 1.0, 1.0) * shadow;
	} else {
		color = vec3(0.0);	
	}

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
