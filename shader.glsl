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
const int MIRROR = 1 << 1;
const int WATER = 1 << 2;
const int SAND = 1 << 3;

const int MAIN_RAY = 0;
const int REFLECTION_RAY = 1;
const int REFRACTION_RAY = 2;

//////////////////////////
// Structs
//////////////////////////

struct Ray {
	vec3 origin;
	vec3 direction;
	int mask;
	int type;
};

struct Hit {
	vec3 position;
	vec3 normal;
	vec3 incident;
	float distance;
	int material;
	int mask;
	int type;
};

struct Renderer {
	Hit hit0;
    Hit hit1;
    Hit hit2;
    Hit hit3;
    Hit hit4;
    Hit hit5;
    Hit hit6;
    Hit hit7;
    
	int num_hits;

	Ray ray0;
    Ray ray1;
    Ray ray2;
    Ray ray3;
    Ray ray4;
    Ray ray5;
    Ray ray6;
    Ray ray7;
    
	int num_rays;

	vec3 pass0;
	vec3 pass1;
	vec3 pass2;
	vec3 pass3;
	vec3 pass4;
	vec3 pass5;
	vec3 pass6;
	vec3 pass7;

	vec3 sun_direction;
};

//////////////////////////
// Functions
//////////////////////////

/* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
vec3 random3(vec3 c) {
	float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
	vec3 r;
	r.z = fract(512.0*j);
	j *= .125;
	r.x = fract(512.0*j);
	j *= .125;
	r.y = fract(512.0*j);
	return r-0.5;
}

/* skew constants for 3d simplex functions */
const float F3 =  0.3333333;
const float G3 =  0.1666667;

/* 3d simplex noise */
float noise(vec3 p) {
	 /* 1. find current tetrahedron T and it's four vertices */
	 /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	 /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
	 
	 /* calculate s and x */
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 /* calculate i1 and i2 */
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 /* x1, x2, x3 */
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 /* 2. find four surflets and store them in d */
	 vec4 w, d;
	 
	 /* calculate surflet weights */
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	 w = max(0.6 - w, 0.0);
	 
	 /* calculate surflet components */
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 /* multiply d by w^4 */
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 /* 3. return the sum of the four surflets */
	 return dot(d, vec4(52.0));
}

float noise(in vec2 p) {
	return noise(vec3(p, 0.0));
}

void mat(inout float a, in float b, inout int mat_a, in int mat_b, in int mask) {
	bool m = (mask & mat_b) > 0;

	mat_a = a < b || m ? mat_a : mat_b;
	a = a < b || m ? a : b;
}

//////////////////////////
// Shapes
//////////////////////////

float plane(in vec3 position, in vec3 normal) {
	return dot(position, normal) / length(normal);
}

float sphere(in vec3 position, in float radius) {
	return length(position) - radius;
}

float box(in vec3 position, in vec3 size) {
	return length(max(abs(position) - size / 2.0, 0.0));
}

//////////////////////////
// Rendering
//////////////////////////

float map(in vec3 position, in int mask, out int material) {
	material = BASE;
	float dist = 1e20;

	float origin = length(position.xy) - 0.5;
	mat(dist, origin, material, BASE, mask);

	float floor_height = noise(position.xz / 10.0) * 0.5;
	float floor = plane(position - vec3(0.0, -10.0 + floor_height, 0.0), vec3(0.0, 1.0, 0.0));
	mat(dist, floor, material, SAND, mask);

	float water_height = noise(position.xz / 50.0 + vec2(pc.time * 0.1, 0.0)) * 6.0;
	water_height += noise(position.xz / 5.0 + vec2(pc.time * 0.1, 0.0)) * 0.2;
	float water = plane(position - vec3(0.0, water_height, 0.0), vec3(0.0, 1.0, 0.0));
	mat(dist, water, material, WATER, mask);

	return dist;
}

vec3 normal(in vec3 position, in int mask)
{
    vec2 e = vec2(1.0,-1.0)*0.5773;
    int material = 0;
    const float eps = 0.0025;
    return normalize(e.xyy * map(position + e.xyy * eps, mask, material) + 
					 e.yyx * map(position + e.yyx * eps, mask, material) + 
					 e.yxy * map(position + e.yxy * eps, mask, material) + 
					 e.xxx * map(position + e.xxx * eps, mask, material) );
}

bool intersect(in Ray ray, out Hit hit) {
	float dist = 0.0;
	int material = 0;

	for (int i = 0; i < MAX_RAY_STEPS; i++) {
		vec3 ray_position = ray.origin + ray.direction * dist;

		float d = map(ray_position, ray.mask, material);

		if (abs(d) < 0.01) {
			break;
		}

		// do stuff, to avoid white dots
		if (dist > 1000.0) {
			return false;
		}

		dist += d;
	}

	hit.position = ray.origin + ray.direction * dist;
	hit.normal = normal(hit.position, ray.mask);
	hit.incident = ray.direction;
	hit.distance = dist;
	hit.material = material;
	hit.mask = ray.mask;
	hit.type = ray.type;
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

void add_ray(inout Renderer renderer, in Ray ray) {
    switch (renderer.num_rays) {
    	case 0: renderer.ray0 = ray; break;
        case 1: renderer.ray1 = ray; break;
        case 2: renderer.ray2 = ray; break;
        case 3: renderer.ray3 = ray; break;
        case 4: renderer.ray4 = ray; break;
        case 5: renderer.ray5 = ray; break;
        case 6: renderer.ray6 = ray; break;
        case 7: renderer.ray7 = ray; break;
    }
    
    renderer.num_rays++;
}

Ray get_ray(inout Renderer renderer, in int ray) {
    switch (ray) {
    	case 0: return renderer.ray0; break;
        case 1: return renderer.ray1; break;
        case 2: return renderer.ray2; break;
        case 3: return renderer.ray3; break;
        case 4: return renderer.ray4; break;
        case 5: return renderer.ray5; break;
        case 6: return renderer.ray6; break;
        case 7: return renderer.ray7; break;
    }
}

void add_hit(inout Renderer renderer, in Hit hit) {
    switch (renderer.num_hits) {
    	case 0: renderer.hit0 = hit; break;
        case 1: renderer.hit1 = hit; break;
        case 2: renderer.hit2 = hit; break;
        case 3: renderer.hit3 = hit; break;
        case 4: renderer.hit4 = hit; break;
        case 5: renderer.hit5 = hit; break;
        case 6: renderer.hit6 = hit; break;
        case 7: renderer.hit7 = hit; break;
    }
    
    renderer.num_hits++;
}

Hit get_hit(inout Renderer renderer, in int hit) {
    switch (hit) {
    	case 0: return renderer.hit0; break;
		case 1: return renderer.hit1; break;
        case 2: return renderer.hit2; break;
        case 3: return renderer.hit3; break;
        case 4: return renderer.hit4; break;
		case 5: return renderer.hit5; break;
        case 6: return renderer.hit6; break;
        case 7: return renderer.hit7; break;
    }
}

void set_pass(inout Renderer renderer, in int pass, in vec3 color) {
	switch (pass) {
		case 0: renderer.pass0 = color; break;
		case 1: renderer.pass1 = color; break;
		case 2: renderer.pass2 = color; break;
		case 3: renderer.pass3 = color; break;
		case 4: renderer.pass4 = color; break;
		case 5: renderer.pass5 = color; break;
		case 6: renderer.pass6 = color; break;
		case 7: renderer.pass7 = color; break;
	}
}

vec3 get_pass(inout Renderer renderer, in int pass) {
	switch (pass) {
		case 0: return renderer.pass0; break;
		case 1: return renderer.pass1; break;
		case 2: return renderer.pass2; break;
		case 3: return renderer.pass3; break;
		case 4: return renderer.pass4; break;
		case 5: return renderer.pass5; break;
		case 6: return renderer.pass6; break;
		case 7: return renderer.pass7; break;
	}
}

void pass(inout Renderer renderer, in Hit hit) {
	switch (hit.material) {
		case MIRROR:
			Ray mirror_ray;
			mirror_ray.origin = hit.position + hit.normal * 0.001;
			mirror_ray.direction = reflect(hit.incident, hit.normal);
			mirror_ray.mask = hit.mask + MIRROR;

			add_ray(renderer, mirror_ray);
        
			break;

		case WATER:
			Ray refraction_ray;
			refraction_ray.origin = hit.position - hit.normal * 0.001;
			refraction_ray.direction = mix(hit.incident, -hit.normal, dot(hit.incident, hit.normal) * -0.1);
			refraction_ray.mask = hit.mask + WATER;
			refraction_ray.type = REFRACTION_RAY;

			Ray reflection_ray;
			reflection_ray.origin = hit.position + hit.normal * 0.001;
			reflection_ray.direction = reflect(hit.incident, hit.normal);
			reflection_ray.mask = hit.mask + WATER;
			reflection_ray.type = REFLECTION_RAY;

			add_ray(renderer, refraction_ray);
			add_ray(renderer, reflection_ray);

			break;

		default:
			break;
	}
}

vec3 material_color(inout Renderer renderer, in int hit_index) {
	Hit hit = get_hit(renderer, hit_index);

	vec3 color = vec3(0.0);

	switch (hit.material) {
		case BASE:
			color = vec3(1.0);
			break;

		case MIRROR:
			color = vec3(0.7);
			break;

		case WATER:
			color = vec3(0.2, 0.4, 0.8);
			break;

		case SAND:
			color = vec3(
				0.7 + 0.1 * pow(noise(hit.position.xz * 20.0), 2.0), 
				0.6 + 0.05 * pow(noise(hit.position.zx * 30.0), 2.0), 
				0.4
			) * (1.0 - clamp(pow(noise(hit.position.xz * 30.0), 8.0), 0.0, 1.0));
			break;

		default:
			break;
	}

	return color;
}

void material_shading(inout Renderer renderer, in int hit_index, in vec3 material, inout vec3 color) {
	Hit hit = get_hit(renderer, hit_index);

	Ray sun_shadow_ray;
	sun_shadow_ray.origin = hit.position + hit.normal * 0.001;
	sun_shadow_ray.direction = renderer.sun_direction;
	sun_shadow_ray.mask = hit.mask;

	float sun_shadow = shadow(sun_shadow_ray, 4.0);	

	float sun_diffuse = clamp(dot(hit.normal, renderer.sun_direction), 0.0, 1.0);	
	float sky_diffuse = clamp(0.4 + 0.6 * dot(hit.normal, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
	float bounce_diffuse = clamp(0.4 + 0.6 * dot(hit.normal, vec3(0.0, -1.0, 0.0)), 0.0, 1.0);

	switch (hit.material) {
		case MIRROR:
			color *= material * sun_diffuse * (0.5 + 0.5 * sun_shadow);
			break;
		case WATER:
			vec3 refraction_color = get_pass(renderer, hit_index + 1);
			vec3 reflection_color = get_pass(renderer, hit_index + 2);

			Hit refraction_hit = get_hit(renderer, hit_index + 1);
			Hit reflection_hit = get_hit(renderer, hit_index + 2);

			color = mix(refraction_color, reflection_color, 0.5 + 0.5 * clamp(0.1 * pow(refraction_hit.distance, 2.0), 0.0, 1.0));

			color *= material;

			color += 0.5 * material * vec3(1.0, 0.9, 0.9) * sun_shadow * sun_diffuse;
			//color += 0.4 * material * vec3(0.3, 0.3, 0.8) * sky_diffuse;
			//color += 0.15 * material * vec3(0.6, 0.3, 0.2) * bounce_diffuse;

			break;
		default:
			color += 1.5 * material * vec3(1.0, 0.9, 0.9) * sun_shadow * sun_diffuse;
			color += 0.4 * material * vec3(0.3, 0.3, 0.8) * sky_diffuse;
			color += 0.15 * material * vec3(0.6, 0.3, 0.2) * bounce_diffuse;
			break;
	}

	color = mix(color, vec3(0.5, 0.7, 0.9), 1.0 - exp(-0.0000001 * pow(hit.distance, 3)));
}

void p(inout Renderer renderer, int i) {
	if (renderer.num_rays > i) {
		Hit hit;
	
		if (intersect(get_ray(renderer, i), hit)) {
			add_hit(renderer, hit);	

			pass(renderer, hit);
		}
	}
}

vec3 m(inout Renderer renderer, int i) {
	vec3 color = vec3(0.0);

	if (renderer.num_hits > i) {
		vec3 material = material_color(renderer, i);
			
		material_shading(renderer, i, material, color);
		set_pass(renderer, i, color);
	} else if (renderer.num_hits == i && renderer.num_rays > renderer.num_hits) {	
		Ray ray = get_ray(renderer, i);

		color = vec3(0.5, 0.7, 0.9) - 0.5 * max(ray.direction.y, 0.0);
		color = mix(color, vec3(0.5, 0.7, 0.9), exp(-10.0 * max(ray.direction.y, 0.0)));
		set_pass(renderer, i, color);
	}

	return color;
}

vec3 render(in Ray ray) {
	Renderer renderer;
	renderer.num_hits = 0;
	renderer.num_rays = 0;
	renderer.sun_direction = normalize(vec3(1.0, 1.0, 0.0));
	add_ray(renderer, ray);

	p(renderer, 0);
	p(renderer, 1);
    p(renderer, 2);
    p(renderer, 3);
    p(renderer, 4);
    p(renderer, 5);
    p(renderer, 6);
    p(renderer, 7);

    m(renderer, 7);
    m(renderer, 6);
    m(renderer, 5);
    m(renderer, 4);
    m(renderer, 3);
    m(renderer, 2);
	m(renderer, 1);
	vec3 color = m(renderer, 0);

	color = pow(color, vec3(0.8, 0.9, 1.0));

	return color;
}

void main() {
	vec3 ray_direction = vec3(v_position, 2.0);
	ray_direction.x *= pc.aspect;

	ray_direction = pc.camera_rotation * normalize(ray_direction);

	Ray ray;
	ray.origin = pc.camera_position;
	ray.direction = ray_direction;
	ray.mask = 0;
	ray.type = MAIN_RAY;

    f_color = vec4(render(ray), 1.0);
}
