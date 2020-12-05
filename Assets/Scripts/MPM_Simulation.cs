using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using UnityEngine;
using UnityEngine.Jobs;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Profiling;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;

public class MPM_Simulation : MonoBehaviour {

    public GameObject prefab;
    public GameObject marker;

    public Mesh instancedMesh;
    public Material instancedMaterial;
    private List<List<Particle>> batches = new List<List<Particle>>();

    struct Particle {
        public float3 x; // position
        public float3 v; // velocity
        public float3x3 C; // affine momentum matrix
        public float3x3 F; // affine momentum matrix
        public float mass;
        public float volume_0; // initial volume
        public bool pinned;
        public Matrix4x4 matrix
        {
            get
            {
                return Matrix4x4.TRS(new Vector3(x.x,x.y,x.z), Quaternion.identity, new Vector3(particle_size, particle_size, particle_size));
            }
        }
    }

    struct Cell {
        public float3 v; // velocity
        public float mass;
        public Vector3Int index3d;
        public bool pinned;
    }
    const float particle_size = 0.5f;
    const int grid_res = 64;
    float3 particle_res = new float3(64,64,16);
    const int num_cells = grid_res * grid_res * grid_res;

    // batch size for the job system. just determined experimentally
    const int division = 16;
    
    // simulation parameters
    const float dt = 0.1f; // timestep
    const float iterations = (int)(1.0f / dt);
    const float gravity = -0.05f;

    const int DISTANCE = 3;
    const float dx = 1.0f / grid_res;
    const float inv_dx = 1.0f;  

    // Lamé parameters for stress-strain relationship
    const float elastic_lambda = 20.0f;
    const float elastic_mu = 10.0f;
    
    NativeArray<Particle> ps; // particles
    NativeArray<Cell> grid;

    float3[] weights = new float3[3];
    static readonly float3x3 identity = math.float3x3(
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            );
    int num_particles;
    List<float3> temp_positions;
    
    #if MOUSE_INTERACTION
    // interaction
    const float mouse_radius = 10;
    bool mouse_down = false;
    float3 mouse_pos;
    #endif
    void createParticles(float3 sp, float3 res) {
        const float spacing = 0.5f;
        //for (float i = - box_x / 2; i < box_x / 2; i += spacing) {
        //    for (float j = - box_y / 2; j < box_y / 2; j += spacing) {
        //        for (float k = -box_z / 2; k < box_z / 2; k += spacing)
        //        {
        //            var pos = math.float3(x + i, y + j, z + k);

        //            temp_positions.Add(pos);
        //        }
        //    }
        //}        
        float3 real_Res = res * spacing;
        for (float i = 0; i < real_Res[0]; i += spacing) {
            for (float j = 0; j < real_Res[1]; j += spacing) {
                for (float k = 0; k < real_Res[2]; k += spacing) {
                    temp_positions.Add(math.float3(i, j, k) + sp);
                }
            }
        }
    }

    void Start () {
        // create particles initial positions
        temp_positions = new List<float3>();
        createParticles(new float3(grid_res / 2, grid_res / 2, grid_res / 2), particle_res);
        num_particles = temp_positions.Count;
        //print(num_particles +"   " +math.pow(particle_res*2, 3));

        ps = new NativeArray<Particle>(num_particles, Allocator.Persistent);



        // initialise particles
        List<Particle> currBatch = new List<Particle>();
        int batchIndexNum = 0;
        for (int i = 0; i < num_particles; ++i) {
            Particle p = new Particle();
            p.x = temp_positions[i];
            p.v = 0;
            p.C = 0;
            p.F = identity;
            
            p.mass = 1.0f;
            //int x = i % particle_res;
            int y = (int)((i / particle_res.x) % particle_res.y);
            //int z = i / (particle_res * particle_res);
            //if (x < 2 || x > particle_res - 3) { p.pinned = 0; }
            // if (y < 2 || y > particle_res - 3) { p.pinned = 0; }
            if (y > particle_res.y - 3)
            {
                p.pinned = true;
                
            } 
            else
            {
                p.pinned = false;
            }
            ps[i] = p;
            
            //GPU splitting particle data
            currBatch.Add(p);
            batchIndexNum++;
            if (batchIndexNum >= 1000)
            {
                batches.Add(currBatch);
                currBatch = new List<Particle>();
                batchIndexNum = 0;
            }



        }

        //Initialize grid:
        grid = new NativeArray<Cell>(num_cells, Allocator.Persistent);

        for (int gx = 0; gx < grid_res; ++gx)
        {
            for (int gy = 0; gy < grid_res; ++gy)
            {
                for (int gz = 0; gz < grid_res; ++gz)
                {
                    // map 3D to 1D index in grid
                    var cell = new Cell();
                    cell.v = 0;
                    cell.index3d = new Vector3Int(gx, gy, gz);
                    int index = gx + (grid_res *gy) + (grid_res * grid_res * gz);
                    cell.pinned = false;
                    grid[index] = cell;
                    

                }
            }
        }
        // ---- begin precomputation of particle volumes
        // MPM course, equation 152 

        //launch a P2G job to scatter particle mass to the grid
        new Job_P2G()
        {
            ps = ps,
            grid = grid,
            num_particles = num_particles
        }.Schedule().Complete();

        for (int i = 0; i < num_particles; ++i) {
            var p = ps[i];
            // quadratic interpolation weights
            float3 cell_idx = math.floor(p.x);
            float3 cell_diff = (p.x - cell_idx) - 0.5f;
            //print(cell_diff);
            weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            weights[1] = 0.75f - math.pow(cell_diff, 2);
            weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);

            //if (i == 6343)
            //{ 
            float density = 0.0f;
            // iterate over neighbouring 3x3 cells
            for (int gx = 0; gx < DISTANCE; ++gx) {
                for (int gy = 0; gy < DISTANCE; ++gy) {
                    for (int gz = 0; gz < DISTANCE; ++gz)
                    {
                        //float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                        float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                        
                        // map 2D to 1D index in grid
                        int cell_index = ((int)cell_idx.x + (gx - 1)) + (grid_res * ((int)cell_idx.y + gy - 1)) + (grid_res * grid_res * ((int)cell_idx.z + (gz - 1))); //??
                        
                        //print("idx" + cell_index + "mass" + grid[cell_index].mass);
                        //var o = Instantiate(marker, grid[cell_index].index3d, Quaternion.identity);
                            //o.transform.localScale = new Vector3(weight, weight, weight);
                        density += grid[cell_index].mass * weight;
                        if (p.pinned)
                        {
                            density *= 100;
                        }
                    }
                }
           // }
            float volume = p.mass / density;
            p.volume_0 = volume;
            
            }
            // per-particle volume estimate has now been computed
            

            ps[i] = p;
        }

        batches.Add(currBatch);

    }

    private void Update() {
#if MOUSE_INTERACTION
        HandleMouseInteraction();
#endif

        Simulate();

        //for(int i=0; i < iterations; i++)
        //{

        //}
        UpdateBatches();
        RenderFrameGPU();
    }
    private void UpdateBatches()
    {
        int batchNumber = 0;
        foreach(var batch in batches)
        {
            int batchSize = batch.Count;
            for(int i = 0; i < batchSize; i++)
            {
                var p = ps[i + batchNumber];
                var bp = batch[i];
                bp = p;
                batch[i] = bp;
            }
            batchNumber += batchSize;
        }


    }
    private void RenderFrameGPU()
    {
        foreach(var batch in batches)
        {
            Graphics.DrawMeshInstanced(instancedMesh, 0, instancedMaterial, batch.Select((a) => a.matrix).ToList());
        }

    }

#if MOUSE_INTERACTION
    void HandleMouseInteraction() {
        mouse_down = false;
        if (Input.GetMouseButton(0)) {
            mouse_down = true;
            var mp = Camera.main.ScreenToViewportPoint(Input.mousePosition);
            mouse_pos = math.float3(mp.x * inv_dx * grid_res, mp.y * grid_res, mp.z * grid_res);
        }
    }
#endif
    void Simulate() {
        Profiler.BeginSample("ClearGrid");
        new Job_ClearGrid() {
            grid = grid
        }.Schedule(num_cells, division).Complete();
        Profiler.EndSample();

        // P2G, first round
        Profiler.BeginSample("P2G");
        new Job_P2G() {
            ps = ps,
            //Fs = Fs,
            grid = grid,
            num_particles = num_particles
        }.Schedule().Complete();
        Profiler.EndSample();
        
        Profiler.BeginSample("Update grid");
        new Job_UpdateGrid() {
            grid = grid
        }.Schedule(num_cells, division).Complete();
        Profiler.EndSample();
        
        Profiler.BeginSample("G2P");
        new Job_G2P() {
            ps = ps,
            //Fs = Fs,
            #if MOUSE_INTERACTION
            mouse_down = mouse_down,
            mouse_pos = mouse_pos,
            #endif
            grid = grid
        }.Schedule(num_particles, division).Complete();
        Profiler.EndSample();
    }

#region Jobs
    [BurstCompile]
    struct Job_ClearGrid : IJobParallelFor {
        public NativeArray<Cell> grid;

        public void Execute(int i) {
            var cell = grid[i];
            cell.mass = 0;
            cell.v = 0;

            grid[i] = cell;
        }
    }
    
    [BurstCompile]
    unsafe struct Job_P2G : IJob {
        public NativeArray<Cell> grid;
        [ReadOnly] public NativeArray<Particle> ps;
        [ReadOnly] public int num_particles;
        
        public void Execute() {
            var weights = stackalloc float3[3];

            for (int i = 0; i < num_particles; ++i) {
                var p = ps[i];

                float3x3 stress = 0;

                // deformation gradient
                var F = p.F;

                var J = math.determinant(F);

                // MPM course, page 46
                var volume = p.volume_0 * J;

                // useful matrices for Neo-Hookean model
                var F_T = math.transpose(F);
                var F_inv_T = math.inverse(F_T);
                var F_minus_F_inv_T = F - F_inv_T;

                // MPM course equation 48
                var P_term_0 = elastic_mu * (F_minus_F_inv_T);
                var P_term_1 = elastic_lambda * math.log(J) * F_inv_T;
                var P = P_term_0 + P_term_1;

                float Dinv = 4 * inv_dx * inv_dx;
                // cauchy_stress = (1 / det(F)) * P * F_T
                // equation 38, MPM course
                stress = (1.0f / J) * Dinv * math.mul(P, F_T);
                
                // (M_p)^-1 = 4, see APIC paper and MPM course page 42
                // this term is used in MLS-MPM paper eq. 16. with quadratic weights, Mp = (1/4) * (delta_x)^2.
                // in this simulation, delta_x = 1, because i scale the rendering of the domain rather than the domain itself.
                // we multiply by dt as part of the process of fusing the momentum and force update for MLS-MPM
                var eq_16_term_0 = -volume * Dinv * stress * dt;

                // quadratic interpolation weights
                uint3 cell_idx = (uint3)(p.x);
                float3 cell_diff = (p.x - cell_idx) - 0.5f;
                //print(cell_diff);
                weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
                weights[1] = 0.75f - math.pow(cell_diff, 2);
                weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);


                // for all surrounding 9 cells
                for (uint gx = 0; gx < DISTANCE; ++gx) {
                    for (uint gy = 0; gy < DISTANCE; ++gy) {
                        for (uint gz = 0; gz < DISTANCE; ++gz)
                        {
                            float weight = weights[gx].x * weights[gy].y * weights[gz].z;

                            uint3 cell_x = math.uint3(cell_idx.x + gx - 1, cell_idx.y + gy - 1, cell_idx.z + gz - 1);
                            float3 cell_dist = (cell_x - p.x) + 0.5f;
                            float3 Q = math.mul(p.C, cell_dist);

                            // scatter mass and momentum to the grid
                            int cell_index = (int)cell_x.x + (grid_res * (int)cell_x.y) + (grid_res * grid_res * (int)cell_x.z);
                            Cell cell = grid[cell_index];

                            // MPM course, equation 172
                            float weighted_mass = weight * p.mass;
                            cell.mass += weighted_mass;

                            // APIC P2G momentum contribution
                            cell.v += weighted_mass * (p.v + Q);

                            // fused force/momentum update from MLS-MPM
                            // see MLS-MPM paper, equation listed after eqn. 28
                            float3 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
                            cell.v += momentum;

                            if (p.pinned)
                            {
                                cell.v = 0;
                                cell.pinned = true;
                                //cell.mass *= 100;
                            }
                            // total update on cell.v is now:
                            // weight * (dt * M^-1 * p.volume * p.stress + p.mass * p.C)
                            // this is the fused momentum + force from MLS-MPM. however, instead of our stress being derived from the energy density,
                            // i use the weak form with cauchy stress. converted:
                            // p.volume_0 * (dΨ/dF)(Fp)*(Fp_transposed)
                            // is equal to p.volume * σ

                            // note: currently "cell.v" refers to MOMENTUM, not velocity!
                            // this gets converted in the UpdateGrid step below.

                            grid[cell_index] = cell;
                        }
                    }
                }
            }
        }
    }

    [BurstCompile]
    struct Job_UpdateGrid : IJobParallelFor {
        public NativeArray<Cell> grid;

        public void Execute(int i) {
            var cell = grid[i];

            if (cell.mass > 0) {
                // convert momentum to velocity, apply gravity
                cell.v /= cell.mass;
                cell.v += dt * math.float3(0, gravity, 0);

                // 'slip' boundary conditions
                int x = i % grid_res;
                int y = (i / grid_res) % grid_res;
                int z = i / (grid_res * grid_res);
                if (x < 2 || x > grid_res - 3) { cell.v.x = 0; }
                if (y < 2 || y > grid_res - 3) { cell.v.y = 0; }
                if (z < 2 || z > grid_res - 3) { cell.v.z = 0; }

                if(cell.pinned)
                {
                    cell.v = 0;
                }
                

                grid[i] = cell;
            }
        }
    }

    [BurstCompile]
    unsafe struct Job_G2P : IJobParallelFor {
        public NativeArray<Particle> ps;
        [ReadOnly] public NativeArray<Cell> grid;

        [ReadOnly] public bool mouse_down;
        [ReadOnly] public float3 mouse_pos;
        
        public void Execute(int i) {
            Particle p = ps[i];

            // reset particle velocity. we calculate it from scratch each step using the grid
            p.v = 0;

            // quadratic interpolation weights
            uint3 cell_idx = (uint3)(p.x);
            float3 cell_diff = ((p.x) - cell_idx) - 0.5f;
            //var weights = stackalloc float3[] {
            //    0.5f * math.pow(0.5f - cell_diff, 2),
            //    0.75f - math.pow(cell_diff, 2), 
            //    0.5f * math.pow(0.5f + cell_diff, 2)
            //};
            var weights = stackalloc float3[3];
            weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
            weights[1] = 0.75f - math.pow(cell_diff, 2);
            weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);
            // constructing affine per-particle momentum matrix from APIC / MLS-MPM.
            // see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
            // below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
            // where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
            //float3x3 B = 0;
            p.C = 0;
            for (uint gx = 0; gx < DISTANCE; ++gx) {
                for (uint gy = 0; gy < DISTANCE; ++gy) {
                    for (uint gz = 0; gz < DISTANCE; ++gz)
                    {
                        float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                        uint3 cell_x = math.uint3(cell_idx.x + gx - 1, cell_idx.y + gy - 1, cell_idx.z + gz - 1);
                        float3 cell_dist = (cell_x - (p.x)) + 0.5f;

                        // scatter mass and momentum to the grid
                        //int cell_index = (int)(((int)cell_x.x + (grid_res * (int)cell_x.y) + (grid_res * grid_res * (int)cell_x.z)) * dx);
                        int cell_index = (int)cell_x.x + (grid_res * (int)cell_x.y) + (grid_res * grid_res * (int)cell_x.z);
                        float3 weighted_velocity = grid[cell_index].v * weight;

                        // APIC paper equation 10, constructing inner term for B
                        //var term = math.float3x3(weighted_velocity * cell_dist.x, weighted_velocity * cell_dist.y, weighted_velocity * cell_dist.z);

                        //B += term;
                        p.C += 4 * inv_dx * math.float3x3(weighted_velocity * cell_dist.x, weighted_velocity * cell_dist.y, weighted_velocity * cell_dist.z);
                        p.v += weighted_velocity;
                    }
                }
            }
            // advect particles

            p.x += p.v * dt;


            // safety clamp to ensure particles don't exit simulation domain
            p.x = math.clamp(p.x, 1, grid_res - 2);

#if MOUSE_INTERACTION
            // mouse interaction
            if (mouse_down) {
                var dist = p.x * inv_dx - mouse_pos;
                if (math.dot(dist, dist) < mouse_radius * mouse_radius) {
                    float norm_factor = (math.length(dist) / mouse_radius);
                    norm_factor = math.pow(math.sqrt(norm_factor), 8);
                    var force = math.normalize(dist) * norm_factor * 0.5f;
                    p.v += force;
                }
            }
#endif

            // deformation gradient update - MPM course, equation 181
            // Fp' = (I + dt * p.C) * Fp
            //var Fp = identity;
            //Fp += dt * p.C;
            p.F = math.mul(identity + (dt* p.C), p.F);

            ps[i] = p;
        }
    }

#endregion

    private void OnDestroy() {
        ps.Dispose();
        grid.Dispose();
    }
}

