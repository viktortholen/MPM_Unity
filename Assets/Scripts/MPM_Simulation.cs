#define PINNED
//#define CUBIC
#define FORCES

using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using UnityEngine;
using UnityEngine.Jobs;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Profiling;

public class MPM_Simulation : MonoBehaviour {

    public GameObject ForceObject;
    const float FORCE_RADIUS = 5.0f;
    public Mesh instancedMesh;
    public Material innerMaterial;
    public Material outerMaterial;

    private List<List<Particle>> batches = new List<List<Particle>>();

    struct Particle {
        public float3 x;
        public float3 v; 
        public float3x3 C;
        public float3x3 F;
        public float mass;
        public float volume;
        public bool pinned;
        public bool isOuter;
        public float dp; //damage
        public Matrix4x4 matrix
        {
            get
            {
                return Matrix4x4.TRS(new Vector3(x.x, x.y, x.z), Quaternion.identity, new Vector3(particle_size, particle_size, particle_size));
            }
        }
    }

    struct Node {
        public float3 v;
        public float mass;
        public float N; //Laplacian
        public float di; //damage
        public Vector3Int index3d;
        public bool pinned;
    }
    //Resolutions
    const float particle_size = 0.7f;
    const int grid_res = 64;
    float3 particle_res = new float3(64,64,8);
    const float spacing = 0.5f;
    const int num_nodes = grid_res * grid_res * grid_res;

    // batch size for CPU parallelization
    const int division = 16;
    
    //Parameters
    const float dt = 0.1f; // timestep
    const float iterations = (int)(1.0f / dt);
    const float gravity = -0.05f;

#if CUBIC
    const int DISTANCE = 2;
    const int Dinv = 3;
#else
    const int DISTANCE = 1;
    const float Dinv = 4 * inv_dx * inv_dx;
#endif

    const float dx = 1.0f / grid_res;
    const float inv_dx = 1.0f;

    const float lambda = 10.0f;
    const float mu = 20.0f; //>= lambda
    
    //Declarations
    NativeArray<Particle> ps; // particle system
    NativeArray<Node> grid;

    static readonly float3x3 identity = math.float3x3( //no identity matrix in math lib.
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            );
    int num_particles;
    List<float3> temp_positions;
    List<int3> temp_indices;

    void createParticles(float3 sp, float3 res) { //create particle box at sp location with resolution res
       
        float3 real_Res = res * spacing;
        for (float i = 0; i < real_Res[0]; i += spacing) {
            for (float j = 0; j < real_Res[1]; j += spacing) {
                for (float k = 0; k < real_Res[2]; k += spacing) {
                    temp_positions.Add(math.float3(i, j, k) + sp);
                    temp_indices.Add((int3)(math.float3(i,j,k) * 2));
                }
            }
        }
    }

    void Start () {
        // create particles initial positions
        temp_positions = new List<float3>();
        temp_indices = new List<int3>();
        createParticles(new float3(grid_res / 2, grid_res / 2, grid_res / 2), particle_res);
        num_particles = temp_positions.Count;

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

            //pinning setup
            var p_idx = temp_indices[i];
            if (p_idx[1] > particle_res.y - 3)  { p.pinned = true; } 
            else { p.pinned = false;}

            //Color setup
            if (p_idx.x > particle_res.x - 2 || p_idx.x < 1 || p_idx.y > particle_res.y - 2 || p_idx.y < 1 || p_idx.z < 1 || p_idx.z > particle_res.z - 2) 
            { p.isOuter = true; }
            else { p.isOuter = false; }

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
        grid = new NativeArray<Node>(num_nodes, Allocator.Persistent);

        for (int gx = 0; gx < grid_res; ++gx)
        {
            for (int gy = 0; gy < grid_res; ++gy)
            {
                for (int gz = 0; gz < grid_res; ++gz)
                {
                    var node = new Node();
                    node.v = 0;
                    node.index3d = new Vector3Int(gx, gy, gz);
                    int index = gx + (grid_res *gy) + (grid_res * grid_res * gz);
                    node.pinned = false;
                    grid[index] = node;
                }
            }
        }

        //launch a P2G job to scatter particle mass to the grid
        new Job_P2G()
        {
            ps = ps,
            grid = grid,
            num_particles = num_particles
        }.Schedule().Complete();


        for (int i = 0; i < num_particles; ++i) {
            var p = ps[i];
            float3 closest_node = math.floor(p.x); //integer 0-res
    
            //if (i == 6343)
            //{ 

            float density = 0.0f;
            // iterate over neighbouring 3x3x3 nodes
            for (int gx = -DISTANCE; gx <= DISTANCE; ++gx)
            {
                for (int gy = -DISTANCE; gy <= DISTANCE; ++gy)
                {
                    for (int gz = -DISTANCE; gz <= DISTANCE; ++gz)
                    {
                        float3 pos = closest_node + new float3(gx, gy, gz);
                        float3 diff = (p.x - (float3)pos) - 0.5f;
                        float weight = Interpolate(diff);

                        int node_idx = GetGridIndex(pos);

                        density += grid[node_idx].mass * weight;
#if PINNED
                        if (p.pinned)
                        {
                            density *= 100;
                        }
#endif
                    }
                }
            }
            float volume = p.mass / density;
            p.volume = volume;

            ps[i] = p;
        }

        batches.Add(currBatch);

    }
    private void Update() {
        Simulate();
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
            //dela upp batch i 2 delar, loopa igenom och splitta materialen
            var outerBatch = new List<Particle>();
            var innerBatch = new List<Particle>();

            foreach(var p in batch)
            {
                if(p.isOuter)
                {
                    outerBatch.Add(p);
                }
                else
                {
                    innerBatch.Add(p);
                }
            }

            Graphics.DrawMeshInstanced(instancedMesh, 0, innerMaterial, innerBatch.Select((a) => a.matrix).ToList());
            Graphics.DrawMeshInstanced(instancedMesh, 0, outerMaterial, outerBatch.Select((a) => a.matrix).ToList());
        }

    }
    public static float Interpolate(float3 pos )
    {
#if CUBIC
        return (InterpolateCubic(pos.x) * InterpolateCubic(pos.y) * InterpolateCubic(pos.z));
#else
        return (InterpolateQuadratic(pos.x) * InterpolateQuadratic(pos.y) * InterpolateQuadratic(pos.z));
#endif
    }
    public static int GetGridIndex(float3 pos)
    {
        return (int)pos.x + (grid_res * (int)pos.y) + (grid_res * grid_res * (int)pos.z);
    }
    public static float InterpolateQuadratic(float p) //quadratic kernel from MPM course
    {
        float x = math.abs(p);

        float w;
        if (x < 0.5f)
            w = (3.0f / 4.0f) - math.pow(x, 2.0f);
        else if (x < 1.5f)
            w = (1.0f / 2.0f) * math.pow((3.0f / 2.0f) - x, 2.0f);
        else w = 0.0f;
        //if (w < 0.0001) return 0;
        return w;
    }
    public static float InterpolateCubic(float p) //cubic kernel from MPM course
    {
        float x = math.abs(p);

        float w;
        if (x < 1.0f)
        {
            w = (1.0f / 2.0f) * math.pow(x, 3.0f) - math.pow(x, 2.0f) + (2.0f / 3.0f); //x * (x * (-x / 6 + 1) - 2) + 4 / 3.0f;
        }
        else if (x < 2.0f)
        {
            w = (-1.0f / 6.0f) * math.pow(x, 3.0f) + math.pow(x, 2.0f) - (2.0f * x) + (4.0f / 3.0f); //Joel Wretborn diva
            //w = (-1.0f / 6.0f) * math.pow(2 - x, 3.0f); //MPMcourse
        }
        else w = 0.0f;

        //if (x < 0.001) return 0;

        return w;
    }

    public static float GetFirstDerivative(float x) //cubic
    {
        //float x_abs = math.abs(x);

        float dw;
        if (x < 1.0)
            dw = math.sign(x)* 1.5f * math.pow(x, 2.0f) - (2.0f*x);
        else if (x < 2.0f)
            dw = -math.sign(x) * 0.5f* math.pow(x, 2.0f) - (2.0f * x) - (2.0f * math.sign(x));
        else dw = 0.0f;
        //if (w < 0.0001) return 0;
        return dw;
    }
    public static float GetSecondDerivative(float x) //cubic
    {
        //float x_abs = math.abs(x);

        float ddw;
        if (x < 1.0)
            ddw = (math.sign(x) * 3.0f * x) - ( math.sign(x) * 2.0f);
        else if (x < 2.0f)
            ddw = ( -math.sign(x) * x )- (math.sign(x) * 2.0f);
        else ddw = 0.0f;
        //if (w < 0.0001) return 0;
        return ddw;
    }
    public static float GetLaplacian(float3 x) //cubic
    {

        return (GetSecondDerivative(x.x) *  InterpolateCubic(x.y)    *  InterpolateCubic(x.z) +
                InterpolateCubic(x.x)    *  GetSecondDerivative(x.y) *  InterpolateCubic(x.z) +
                InterpolateCubic(x.x)    *  InterpolateCubic(x.y)    *  GetSecondDerivative(x.z));
    }
    
    void Simulate() {  //setup all jobs parallel and send the grid and ps
        Profiler.BeginSample("ClearGrid");
        new Job_ClearGrid() {
            grid = grid
        }.Schedule(num_nodes, division).Complete();
        Profiler.EndSample();

        Profiler.BeginSample("P2G");
        new Job_P2G() {
            ps = ps,
            grid = grid,
            num_particles = num_particles
        }.Schedule().Complete();
        Profiler.EndSample();
        
        Profiler.BeginSample("Update grid");
        new Job_UpdateGrid() {
            grid = grid
        }.Schedule(num_nodes, division).Complete();
        Profiler.EndSample();
        
        Profiler.BeginSample("G2P");
        new Job_G2P() {
            ps = ps,
            force_pos = (float3)ForceObject.transform.position,
            grid = grid
        }.Schedule(num_particles, division).Complete();
        Profiler.EndSample();
    }

#region Jobs 
    //Jobs are done in parallel with CPU - unsafe is required for math.log which can be unsafe if det(F)
    //is negative (wont crash the application, particle will be deleted instead)

    [BurstCompile]
    struct Job_ClearGrid : IJobParallelFor {
        public NativeArray<Node> grid;

        public void Execute(int i) {
            var node = grid[i];
            node.mass = 0;
            node.v = 0;

            grid[i] = node;
        }
    }
    
    [BurstCompile]
    unsafe struct Job_P2G : IJob {
        public NativeArray<Node> grid;
        [ReadOnly] public NativeArray<Particle> ps;
        [ReadOnly] public int num_particles;
        
        public void Execute() {

            for (int i = 0; i < num_particles; ++i) {
                var p = ps[i];

                float3x3 stress = 0;

                // deformation gradient
                var F = p.F;

                var J = math.determinant(F);

                // MPM course, page 46
                var V = p.volume * J;

                // MPM course equation 48
                var FT = math.transpose(F);
                var FinvT = math.inverse(FT);
                var P0 = mu * (F - FinvT);
                var P1 = lambda * math.log(J) * FinvT;
                var P = P0 + P1;


                // equation 38, MPM course
                stress = (1.0f / J) * math.mul(P, FT);

                var force_term = -V * Dinv * stress * dt;

                int3 closest_node = (int3)(p.x);

                for (int gx = -DISTANCE; gx <= DISTANCE; ++gx)
                {
                    for (int gy = -DISTANCE; gy <= DISTANCE; ++gy)
                    {
                        for (int gz = -DISTANCE; gz <= DISTANCE; ++gz)
                        {
                            float3 pos = closest_node + new float3(gx, gy, gz);
                            float3 diff = (p.x - (float3)pos) - 0.5f; //-0.5 to interpolate symmetrical
                            float weight = Interpolate(diff);

                            float3 dist = ((float3)pos - p.x) + 0.5f; //+0.5 to get real distance as -0.5 in interpolation

                            int node_idx = GetGridIndex(pos);
                            Node node = grid[node_idx];

                            // MPM mass equation 172
                            float weighted_mass = weight * p.mass;
                            node.mass += weighted_mass;

                            // Momentum MPM 178
                            node.v += weighted_mass * (p.v + math.mul(p.C, dist));

                            // force equation MLS
                            float3 force = math.mul(force_term * weight, dist);
                            node.v += force;
#if PINNED
                            if (p.pinned)
                            {
                                node.v = 0;
                                node.pinned = true;
                            }
#endif

                            //Aniso: Compute and store Laplacians (Explicit)
                            //node.di += weight * p.dp;
                            //node.N = GetLaplacian(diff);
                            //p.dp += node.di * node.N;

                            grid[node_idx] = node;
                        }
                    }
                }
                //Update Anisotropic Damage (Explicit only)*******************************************

                //Compute geometric resistance: D
                //Compute driving force:
                //Get un-degraded Cauchy stress
                //Get eigenvalues and eigenvectors
                //Calculate stress (new way)
                //constructStructuralTensor
                //Calculate elastic thingy (instead of eq 48)
                //apply critical stress conditions and set damage

                //************************************************************************************
            }
        }
    }

    [BurstCompile]
    struct Job_UpdateGrid : IJobParallelFor {
        public NativeArray<Node> grid;

        public void Execute(int i) {
            var node = grid[i];

            if (node.mass > 0.0f) {
                // convert momentum to velocity, apply gravity
                node.v /= node.mass;
                node.v += dt * math.float3(0, gravity, 0);


                //boundaries
                int x = i % grid_res;
                int y = (i / grid_res) % grid_res;
                int z = i / (grid_res * grid_res);
                if (x < 3 || x > grid_res - 2) { node.v.x = 0; }
                if (y < 3 || y > grid_res - 2) { node.v.y = 0; }
                if (z < 3 || z > grid_res - 2) { node.v.z = 0; }
#if PINNED
                if(node.pinned)
                {
                    node.v = 0;
                }
#endif
                grid[i] = node;
            }
        }
    }

    [BurstCompile]
    unsafe struct Job_G2P : IJobParallelFor {
        public NativeArray<Particle> ps;
        [ReadOnly] public NativeArray<Node> grid;
        [ReadOnly] public float3 force_pos;
        
        public void Execute(int i) {
            Particle p = ps[i];

            // reset particle velocity. we calculate it from scratch each step using the grid
            p.v = 0;

            int3 closest_node = (int3)(p.x);

            float3x3 B = 0;
            for (int gx = -DISTANCE; gx <= DISTANCE; ++gx)
            {
                for (int gy = -DISTANCE; gy <= DISTANCE; ++gy)
                {
                    for (int gz = -DISTANCE; gz <= DISTANCE; ++gz)
                    {
                        float3 pos = closest_node + new float3(gx, gy, gz);
                        float3 diff = (p.x - (float3)pos) - 0.5f;
                        float weight = Interpolate(diff);
                        float3 dist = ((float3)pos - p.x) + 0.5f;

                        int node_idx = GetGridIndex(pos);
                        float3 weighted_velocity = grid[node_idx].v * weight;

                        // inner term for B
                        var term = math.float3x3(weighted_velocity * dist.x, weighted_velocity * dist.y, weighted_velocity * dist.z);
                        B += term;
                        p.v += weighted_velocity;
                    }
                }
            }
            p.C = B * Dinv;
            p.x += p.v * dt;
            p.x = math.clamp(p.x, 3, grid_res - 3);

#if FORCES
            var force_dist = p.x * inv_dx - force_pos;
            if(math.dot(force_dist, force_dist) < FORCE_RADIUS * FORCE_RADIUS)
            {
                float norm = math.pow(math.sqrt((math.length(force_dist) / FORCE_RADIUS)), 8);
                float3 force = math.normalize(force_dist) * norm * 1.0f;
                p.v += force;
            }
#endif  
            // equation 181
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

