import os
import json
import torch


def search_thoroughly_enough_and_save_prompts(
    diffuser,
    initial_erase_concept,
    initializer_token,
    train_data_dir,
    train_method,
    lr,
    ti_lr,
    negative_guidance,
    iterations,
    n_iterations,
    device,
    ti_max_train_steps,
    learnable_property,
    output_dir,
    generic_prompt,
    center_crop=False,
    prompt_save_name="adv_prompt_log.json",
):
    """
    Runs STEREO Stage 1 and saves adversarial placeholder-token prompts in a clean format.

    What this function saves:
    - current concept being erased at each iteration
    - new placeholder token discovered at each iteration
    - prompt built from current concept
    - prompt built from new token
    - cumulative token prompts discovered so far
    - paths to saved erased UNet and attacked text encoder checkpoints

    Important:
    This saves token-based adversarial prompts of the form:
        f"{generic_prompt} {token}"
    These are placeholder-token prompts, not rich natural-language adversarial prompts.
    """

    os.makedirs(output_dir, exist_ok=True)

    current_concept = initial_erase_concept
    saved_tokens = {}
    adv_prompt_log = {}

    def dedup_preserve_order(items):
        return list(dict.fromkeys(items))

    prompt_save_path = os.path.join(output_dir, prompt_save_name)

    for iteration in range(n_iterations):
        placeholder_token = generate_unique_placeholder_token(saved_tokens, iteration)
        saved_tokens[str(iteration)] = placeholder_token

        erased_weights_path = os.path.join(output_dir, f"erased_unet_iteration_{iteration}.pt")
        attack_model_path = os.path.join(output_dir, f"ci_attack_text_encoder_iteration_{iteration}.pt")

        print(
            f"\n{'=' * 30} Iteration {iteration + 1}/{n_iterations} {'=' * 30}"
        )
        print(
            f"Erasing concept: {current_concept} -> "
            f"Placeholder token: '{placeholder_token}' "
            f"(initialized from '{initializer_token}')"
        )

        # 1. Erase current concept
        diffuser = train_erasing(
            erase_concept=current_concept,
            erase_from=current_concept,
            train_method=train_method,
            iterations=iterations,
            negative_guidance=negative_guidance,
            lr=lr,
            save_path=erased_weights_path,
            diffuser=diffuser,
            device=device,
        )
        print(f"Erased weights saved to {erased_weights_path}")

        # 2. Attack via textual inversion
        diffuser.unet.load_state_dict(torch.load(erased_weights_path, map_location=device))
        torch.cuda.empty_cache()

        diffuser = train_concept_inversion(
            diffuser=diffuser,
            placeholder_token=placeholder_token,
            initializer_token=initializer_token,
            train_data_dir=train_data_dir,
            lr=ti_lr,
            save_path=attack_model_path,
            device=device,
            max_train_steps=ti_max_train_steps,
            learnable_property=learnable_property,
            scale_lr=True,
            iteration=iteration,
            num_iterations=n_iterations,
            center_crop=center_crop,
        )
        print(f"Attacked model with placeholder '{placeholder_token}' saved to {attack_model_path}")

        # 3. Build prompt records BEFORE mutating current_concept
        prompt_current_concept = f"{generic_prompt} {current_concept}"
        prompt_new_token = f"{generic_prompt} {placeholder_token}"
        all_token_prompts_so_far = dedup_preserve_order(
            [f"{generic_prompt} {tok}" for tok in saved_tokens.values()]
        )

        adv_prompt_log[f"iteration_{iteration}"] = {
            "iteration": iteration,
            "current_concept": current_concept,
            "new_placeholder_token": placeholder_token,
            "initializer_token": initializer_token,
            "prompt_current_concept": prompt_current_concept,
            "prompt_new_token": prompt_new_token,
            "all_token_prompts_so_far": all_token_prompts_so_far,
            "saved_tokens_so_far": dict(saved_tokens),
            "erased_weights_path": erased_weights_path,
            "attack_model_path": attack_model_path,
        }

        # Save intermediate JSON every iteration
        with open(prompt_save_path, "w", encoding="utf-8") as f:
            json.dump(adv_prompt_log, f, indent=2, ensure_ascii=False)

        print(f"Saved adversarial prompt log to {prompt_save_path}")

        # 4. Run inference
        print(
            f"Generating images for current_concept='{current_concept}', "
            f"placeholder_token='{placeholder_token}', generic_prompt='{generic_prompt}'"
        )
        inference_and_save(
            generic_prompt=generic_prompt,
            prompt=current_concept,
            placeholder_token=placeholder_token,
            saved_tokens=saved_tokens,
            iteration=iteration,
            output_dir=output_dir,
            device=device,
        )

        # Move to next concept after logging
        current_concept = placeholder_token

        print(f"{'=' * 30} Iteration {iteration + 1} complete {'=' * 30}\n")
        torch.cuda.empty_cache()

    return diffuser, saved_tokens, adv_prompt_log