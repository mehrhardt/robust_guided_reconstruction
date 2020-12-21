function affmap = compute_affine_map(moving,fixed,maxIt)

    xWorldLimits = [-1 1];
    yWorldLimits = [-1 1];
    
    Rmoving = imref2d(size(moving),xWorldLimits,yWorldLimits);
    Rfixed = imref2d(size(moving),xWorldLimits,yWorldLimits);

    [optimizer, metric] = imregconfig('multimodal');
    optimizer.InitialRadius = 0.0019;
    optimizer.Epsilon = 1.5e-5; 
    optimizer.GrowthFactor = 1.01;
    optimizer.MaximumIterations = maxIt;
    
    tform = imregtform(moving, Rmoving, fixed, Rfixed, 'affine', optimizer, metric);
    affmap = tform.T;
end
