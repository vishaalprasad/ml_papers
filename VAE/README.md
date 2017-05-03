Variational Autoencoders (VAEs)

This is an implementation of VAEs, as described in Kingma and Welling (2013).

Credit to Jaan Altosaar (https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) for the elaboration on both the autoencoder & Variational Bayesian explanations. This explanation is very much his tutorial repackaged in a way more digestible for me.

VAEs are generative models that lie in the intersection of Variational Bayesian methods and neural networks (i.e. autoencoders). As my background is with neural networks, I will attempt to explain this from a Bayesian point of view to further my learning.

VAEs can be seen as the simple Bayesian graphical model: z -> x. x is our data and z is the set of latent variables. What this means is that the data x is dependent on some underlying set of latent variables z. That also means if we want to generate data, we first sample from p(z) and then, dependent on that, we actually form the image by sampling from p(x|z). To make this concrete, take an example of handwritten digits (0-9). Our latent variables might simply be a digit 0-9 (though many more latent variables may underly the data: rotation, whether the 2 has a loop, etc). If we are to form a sample image, first we would sample for our latent factors via p(z), which might give us the digit 5. Based on that "5" that we sampled, the model would generate an image that looks like a 5 via p(x|z).

If we are to do inferencing, we wish to fit the posterior. Namely, we want to find the most likely latent variables given the data, i.e. maximize p(z|x). From Bayes rule, we know that P(z|x) = p(x|z)p(z)/p(x). We know how to find p(x|z) and p(z), but what about p(x)? From the law of total probability, we get p(x) = ∫p(x|z)p(z)dz (assume z is continuous). 

And now we arise at a problem. Evaluating this integral means going over every configuration of z, which requires exponential time over the latent factors. So we need to do something smarter. We'll instead approximate the true posterior using some family of distributions, denoted q<sub>λ</sub>(z|x). All λ does is tell us a particular configuration in that family. For example, suppose we choose the Gaussian as our approximation; then we might have λ<sub>x</sub>=(μ<sub>x</sub>,σ<sup>2</sup><sub>x</sub>) for some specific training example x. So all this says it that we give every node the same probability function, but they might have different parameters.

We choose q<sub>λ</sub>(z|x) to be Gaussian. Now what? Our goal was to find the best set of parameters λ to fit the data X, but what exactly does "best" mean? We originally chose q<sub>λ</sub>(z|x) as a replacement for the posterior p(z|x), so why don't we measure the Kullback-Liebler (KL) divergence between the two distributions?

KL(q<sub>λ</sub>(z|x) || p(z|x)) = E<sub>q</sub>[log q<sub>λ</sub>(z|x)] - E<sub>q</sub>[log p(x,z)] + p(x).

We want to optimize q<sub>λ</sub>(z|x) based on the above, i.e. find argmax<sub>λ</sub>KL(q<sub>λ</sub>(z|x) || p(z|x)). the problem is that we have p(x) back in this expression -- this is the thing we are trying to get around.

If we define ELBO(λ) =  E<sub>q</sub>[log p(x,z)] - E<sub>q</sub>[log q<sub>λ</sub>(z|x)], then combined with the definition of KL divergence, we get:

log p(x) = ELBO(λ) + KL(q<sub>λ</sub>(z|x) || p(z|x)).

p(x) is a constant with respect to λ, and so we simply need to maximize ELBO(λ) to minimize
