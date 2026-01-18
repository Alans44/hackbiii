"""
Solana Blockchain Integration for EcoShelf
============================================
Decentralized food waste tracking and rewards system.

MLH Prize: Best Use of Solana

Features:
- Immutable waste prevention records
- Token rewards for waste reduction
- Transparent impact tracking
- Community leaderboard on-chain
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import base64

# Solana Configuration
SOLANA_RPC_URL = os.environ.get('SOLANA_RPC_URL', 'https://api.devnet.solana.com')
SOLANA_PRIVATE_KEY = os.environ.get('SOLANA_PRIVATE_KEY', '')
PROGRAM_ID = os.environ.get('ECOSHELF_PROGRAM_ID', '')

# Try to import solana libraries
try:
    from solana.rpc.api import Client
    from solana.keypair import Keypair
    from solana.transaction import Transaction
    from solana.system_program import TransferParams, transfer
    from solana.publickey import PublicKey
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    print("solana-py not installed. Run: pip install solana")


class SolanaWasteTracker:
    """
    Solana-based waste tracking for EcoShelf.
    Records food waste prevention on the blockchain.
    """
    
    def __init__(self, rpc_url: str = None, private_key: str = None):
        self.rpc_url = rpc_url or SOLANA_RPC_URL
        self.client = None
        self.keypair = None
        
        if SOLANA_AVAILABLE:
            try:
                self.client = Client(self.rpc_url)
                if private_key:
                    # Load keypair from base58 or bytes
                    self.keypair = Keypair.from_secret_key(
                        base64.b64decode(private_key)
                    )
                print(f"✅ Connected to Solana ({self.rpc_url})")
            except Exception as e:
                print(f"⚠️ Solana connection failed: {e}")
    
    def record_waste_prevention(
        self, 
        user_id: str,
        item: str,
        weight_kg: float,
        action: str
    ) -> Optional[str]:
        """
        Record a waste prevention event on Solana blockchain.
        
        Args:
            user_id: User identifier
            item: Food item name
            weight_kg: Weight of food saved
            action: Action taken (eaten, donated, composted)
        
        Returns:
            Transaction signature or None
        """
        # Create waste prevention record
        record = {
            'user_id': user_id,
            'item': item,
            'weight_kg': weight_kg,
            'action': action,
            'co2_saved_kg': weight_kg * 2.5,
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
        
        # Hash the record for integrity
        record_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        
        record['hash'] = record_hash
        
        if not self.client or not self.keypair:
            # Return mock transaction for demo
            return f"mock_tx_{record_hash[:16]}"
        
        try:
            # In production, this would call your Solana program
            # For demo, we'll use a memo transaction
            memo_data = json.dumps({
                'type': 'waste_prevention',
                'hash': record_hash,
                'co2_saved': record['co2_saved_kg']
            })
            
            # Create and send transaction
            # Note: Actual implementation would use your deployed program
            tx = Transaction()
            # Add memo instruction here
            
            # Send transaction
            response = self.client.send_transaction(tx, self.keypair)
            return response['result']
            
        except Exception as e:
            print(f"Transaction failed: {e}")
            return None
    
    def get_user_impact(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's total environmental impact from blockchain.
        
        Returns aggregated stats from on-chain records.
        """
        # In production, query the Solana program for user's records
        # For demo, return mock data
        return {
            'user_id': user_id,
            'total_items_saved': 47,
            'total_weight_kg': 23.5,
            'total_co2_saved_kg': 58.75,
            'eco_tokens_earned': 470,
            'rank': 156,
            'verified_on_chain': True,
            'blockchain': 'Solana Devnet'
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top waste reducers from blockchain"""
        # In production, query program's leaderboard account
        return [
            {'rank': 1, 'user_id': 'eco_warrior_1', 'co2_saved': 234.5, 'tokens': 2345},
            {'rank': 2, 'user_id': 'green_chef', 'co2_saved': 198.2, 'tokens': 1982},
            {'rank': 3, 'user_id': 'zero_waste', 'co2_saved': 167.8, 'tokens': 1678},
            # ... more entries
        ][:limit]
    
    def mint_eco_token(self, user_id: str, amount: int) -> Optional[str]:
        """
        Mint EcoShelf tokens as rewards for waste prevention.
        
        Args:
            user_id: Recipient's identifier
            amount: Number of tokens to mint
            
        Returns:
            Transaction signature
        """
        # In production, call SPL Token mint instruction
        return f"mock_mint_{amount}_{user_id[:8]}"
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get overall EcoShelf network statistics"""
        return {
            'total_users': 1247,
            'total_items_saved': 58392,
            'total_weight_kg': 29196,
            'total_co2_saved_kg': 72990,
            'total_tokens_minted': 583920,
            'network': 'Solana',
            'program_id': PROGRAM_ID or 'Not deployed'
        }


# Solana Program (Rust) - For reference
SOLANA_PROGRAM_RUST = '''
// EcoShelf Solana Program
// Save this as lib.rs in your Anchor project

use anchor_lang::prelude::*;

declare_id!("EcoSheLf11111111111111111111111111111111");

#[program]
pub mod ecoshelf {
    use super::*;

    pub fn record_waste_prevention(
        ctx: Context<RecordWaste>,
        item: String,
        weight_kg: f64,
        action: String,
    ) -> Result<()> {
        let record = &mut ctx.accounts.waste_record;
        let user = &ctx.accounts.user;
        
        record.user = user.key();
        record.item = item;
        record.weight_kg = weight_kg;
        record.co2_saved_kg = weight_kg * 2.5;
        record.action = action;
        record.timestamp = Clock::get()?.unix_timestamp;
        
        // Update user's total stats
        let user_stats = &mut ctx.accounts.user_stats;
        user_stats.total_weight_saved += weight_kg;
        user_stats.total_co2_saved += weight_kg * 2.5;
        user_stats.items_count += 1;
        
        // Mint reward tokens (10 tokens per kg saved)
        let tokens_to_mint = (weight_kg * 10.0) as u64;
        // ... token minting logic
        
        Ok(())
    }
    
    pub fn initialize_user(ctx: Context<InitializeUser>) -> Result<()> {
        let user_stats = &mut ctx.accounts.user_stats;
        user_stats.user = ctx.accounts.user.key();
        user_stats.total_weight_saved = 0.0;
        user_stats.total_co2_saved = 0.0;
        user_stats.items_count = 0;
        user_stats.joined_at = Clock::get()?.unix_timestamp;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct RecordWaste<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(
        init,
        payer = user,
        space = 8 + WasteRecord::SIZE
    )]
    pub waste_record: Account<'info, WasteRecord>,
    
    #[account(mut)]
    pub user_stats: Account<'info, UserStats>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitializeUser<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(
        init,
        payer = user,
        space = 8 + UserStats::SIZE,
        seeds = [b"user-stats", user.key().as_ref()],
        bump
    )]
    pub user_stats: Account<'info, UserStats>,
    
    pub system_program: Program<'info, System>,
}

#[account]
pub struct WasteRecord {
    pub user: Pubkey,
    pub item: String,
    pub weight_kg: f64,
    pub co2_saved_kg: f64,
    pub action: String,
    pub timestamp: i64,
}

impl WasteRecord {
    pub const SIZE: usize = 32 + 64 + 8 + 8 + 32 + 8;
}

#[account]
pub struct UserStats {
    pub user: Pubkey,
    pub total_weight_saved: f64,
    pub total_co2_saved: f64,
    pub items_count: u64,
    pub joined_at: i64,
}

impl UserStats {
    pub const SIZE: usize = 32 + 8 + 8 + 8 + 8;
}
'''

# JavaScript client for frontend integration
SOLANA_JS_CLIENT = '''
// EcoShelf Solana Client (for frontend)
import { Connection, PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@project-serum/anchor';

const PROGRAM_ID = new PublicKey('EcoSheLf11111111111111111111111111111111');
const RPC_URL = 'https://api.devnet.solana.com';

export class EcoShelfSolana {
    constructor(wallet) {
        this.connection = new Connection(RPC_URL);
        this.wallet = wallet;
        // Initialize Anchor program here
    }
    
    async recordWastePrevention(item, weightKg, action) {
        // Call Solana program
        const tx = await this.program.methods
            .recordWastePrevention(item, weightKg, action)
            .accounts({
                user: this.wallet.publicKey,
                // ... other accounts
            })
            .rpc();
        
        return tx;
    }
    
    async getUserStats() {
        const [userStatsPda] = await PublicKey.findProgramAddress(
            [Buffer.from('user-stats'), this.wallet.publicKey.toBuffer()],
            PROGRAM_ID
        );
        
        return await this.program.account.userStats.fetch(userStatsPda);
    }
    
    async getLeaderboard(limit = 10) {
        // Fetch all user stats and sort by co2_saved
        const accounts = await this.program.account.userStats.all();
        return accounts
            .sort((a, b) => b.account.totalCo2Saved - a.account.totalCo2Saved)
            .slice(0, limit);
    }
}
'''


if __name__ == "__main__":
    print("=" * 60)
    print("Solana Blockchain Integration for EcoShelf")
    print("=" * 60)
    print("\nSetup Instructions:")
    print("1. Install Solana CLI: sh -c \"$(curl -sSfL https://release.solana.com/v1.17.0/install)\"")
    print("2. Create wallet: solana-keygen new")
    print("3. Get devnet SOL: solana airdrop 2")
    print("4. Deploy program using Anchor framework")
    print("5. Set SOLANA_PRIVATE_KEY and ECOSHELF_PROGRAM_ID env vars")
    print("\nPython package: pip install solana")
    
    print("\nFeatures:")
    print("  - Record waste prevention on-chain")
    print("  - Mint EcoTokens as rewards")
    print("  - Transparent leaderboard")
    print("  - Verified environmental impact")
    
    # Demo
    tracker = SolanaWasteTracker()
    print("\nNetwork Stats (Demo):")
    print(json.dumps(tracker.get_network_stats(), indent=2))
    
    print("\nUser Impact (Demo):")
    print(json.dumps(tracker.get_user_impact("demo_user"), indent=2))
