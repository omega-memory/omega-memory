declare module "three" {
  export class Group {
    add(object: unknown): void;
  }
  export class Mesh {
    scale: { set(x: number, y: number, z: number): void };
    position: { set(x: number, y: number, z: number): void };
    constructor(geometry: unknown, material: unknown);
  }
  export class SphereGeometry {
    constructor(radius?: number, widthSegments?: number, heightSegments?: number);
  }
  export class RingGeometry {
    constructor(innerRadius?: number, outerRadius?: number, thetaSegments?: number);
  }
  export class OctahedronGeometry {
    constructor(radius?: number, detail?: number);
  }
  export class TorusGeometry {
    constructor(radius?: number, tube?: number, radialSegments?: number, tubularSegments?: number);
  }
  export class IcosahedronGeometry {
    constructor(radius?: number, detail?: number);
  }
  export class MeshBasicMaterial {
    constructor(params?: {
      color?: Color | number;
      transparent?: boolean;
      opacity?: number;
      blending?: number;
      depthWrite?: boolean;
      side?: number;
      wireframe?: boolean;
    });
  }
  export class Color {
    constructor(hex: number | string);
  }
  export const AdditiveBlending: number;
  export const DoubleSide: number;
}
